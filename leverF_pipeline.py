from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


@dataclass
class Config:
    rsna_root: str = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
    manifest_csv: str = ""
    nih_root: str = ""
    kermany_root: str = ""
    cache_dir: str = "/tmp/png_cache"
    archive_path: str = "/kaggle/working/png_cache.zip"
    out_root: str = "/kaggle/working/rsna_leverF_results"

    cache_size: int = 256
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    head_epochs: int = 3
    full_epochs: int = 24
    head_lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    stage1: str = "none"
    kermany_head_epochs: int = 5
    kermany_full_epochs: int = 10
    kermany_head_lr: float = 1e-3
    kermany_backbone_lr: float = 1e-4
    kermany_normal_limit: int = 1501
    kermany_calibration_fraction: float = 0.1
    nih_opacity_findings: tuple = ("Infiltration", "Consolidation", "Pneumonia")
    nih_subset_per_class: int = 25000
    nih_calibration_fraction: float = 0.1
    nih_exclude_csv: str = ""
    nih_head_epochs: int = 5
    nih_full_epochs: int = 10

    loss: str = "focal_smoothing"
    focal_gamma: float = 2.0
    focal_alpha_positive: float = 0.35
    label_smoothing: float = 0.05

    warmup_frac: float = 0.05
    dropout: float = 0.3
    ema_decay: float = 0.999
    select_best_epoch: bool = True
    patience: int = 6

    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.01
    threshold_bootstrap_iterations: int = 500
    bootstrap_iterations: int = 2000

    refit_on_train_plus_calibration: bool = True
    tta: tuple = ("orig", "hflip")

    snapshot_every_min: int = 45
    snapshot_path: str = "/kaggle/working/leverF_snapshot.zip"
    resume_from: str = "auto"
    force_retrain: bool = False

    arm: str = "stage1none"
    backbone: str = "efficientnet_b4"
    seed: int = 42

    debug: bool = False
    debug_per_class: int = 160
    debug_stage1_per_class: int = 120


POSITIVE_CLASS = "Lung Opacity"
BACKBONES = ("efficientnet_b4", "densenet121")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def run_dir_for(cfg: Config) -> Path:
    return Path(cfg.out_root) / f"leverF__{cfg.arm}__{cfg.backbone}__seed{cfg.seed}"


def make_snapshot(cfg: Config) -> Path | None:
    return None
    import zipfile

    out = Path(cfg.out_root)
    if not out.exists():
        return None

    dst = Path(cfg.snapshot_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    tmp = dst.with_name(f"{dst.name}.{_os.getpid()}.partial")

    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out.parent))
                n += 1
    try:
        tmp.replace(dst)
    except FileNotFoundError:
        print('  [snapshot] skipped (concurrent writer)', flush=True)
        return None

    print(f"  [snapshot] {n} files, {dst.stat().st_size / 1e6:.1f} MB -> {dst}", flush=True)
    return dst


def start_snapshot_thread(cfg: Config):
    import threading

    if cfg.snapshot_every_min <= 0:
        return None

    stop = threading.Event()

    def loop():
        while not stop.wait(cfg.snapshot_every_min * 60):
            try:
                make_snapshot(cfg)
            except Exception as exc:
                print(f"  [snapshot] failed: {exc}", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    print(f"  [snapshot] every {cfg.snapshot_every_min} min -> {cfg.snapshot_path}")
    return stop


def autolocate(cfg: Config) -> None:
    roots = [Path(r) for r in ("/kaggle/input", "/kaggle/working") if Path(r).is_dir()]

    if not Path(cfg.manifest_csv or "/nonexistent").is_file():
        hits = [p for root in roots for p in root.glob("**/rsna_split_manifest.csv")]
        if not hits:
            raise SystemExit(
                "Could not find rsna_split_manifest.csv.\n"
                "Add the dataset containing it as a notebook input, or pass "
                "--manifest-csv with the full path.")
        cfg.manifest_csv = str(hits[0])
        print(f"found manifest: {cfg.manifest_csv}")

    if cfg.stage1 == "nih" and not Path(cfg.nih_root or "/nonexistent").is_dir():
        hits = [p.parent for root in roots for p in root.glob("**/Data_Entry_2017*.csv")]
        if hits:
            cfg.nih_root = str(hits[0])
            print(f"found NIH CXR-14: {cfg.nih_root}")

    if cfg.stage1 == "kermany" and not Path(cfg.kermany_root or "/nonexistent").is_dir():
        hits = [p for root in roots for p in root.glob("**/chest_xray/train")]
        if hits:
            cfg.kermany_root = str(hits[0].parent)
            print(f"found Kermany: {cfg.kermany_root}")

    cache = Path(cfg.cache_dir)
    if cache.is_dir() and any(cache.glob("*.png")):
        return

    hits = [p for root in roots for p in root.glob("**/png_cache")
            if p.is_dir() and any(p.glob("*.png"))]
    if hits:
        cfg.cache_dir = str(hits[0])
        print(f"found image cache: {cfg.cache_dir}")
        return

    archives = [p for root in roots for p in root.glob("**/png_cache.zip")]
    if archives:
        unpack_cache_archive(archives[0], Path("/tmp/png_cache"))
        cfg.cache_dir = "/tmp/png_cache"


def unpack_cache_archive(src: Path, dest: Path) -> None:
    import time
    import zipfile

    if dest.is_dir() and any(dest.glob("*.png")):
        print(f"image cache already unpacked at {dest}")
        return

    print(f"unpacking image cache: {src.name} -> {dest} (once per session)", flush=True)
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest.parent)

    n = len(list(dest.glob("*.png")))
    if n == 0:
        raise SystemExit(
            f"{src.name} unpacked but no PNGs landed in {dest}. The archive probably has "
            f"a different internal folder name.")
    print(f"unpacked {n} images in {time.time() - t0:.0f}s", flush=True)


_RESUME_DONE = False


def find_snapshot(cfg: Config) -> Path | None:
    setting = (cfg.resume_from or "none").strip()
    if setting.lower() in {"none", "off"}:
        return None
    if setting.lower() != "auto":
        src = Path(setting)
        if not src.exists():
            raise SystemExit(f"--resume-from not found: {src}")
        return src

    roots = [Path(r) for r in ("/kaggle/input", "/kaggle/working") if Path(r).is_dir()]
    found = [p for root in roots for p in root.glob("**/leverF_snapshot*.zip") if p.is_file()]
    found = [p for p in found if p != Path(cfg.snapshot_path)]
    if not found:
        print("resume: no leverF_snapshot.zip in /kaggle/input - starting clean")
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def unpack_resume(cfg: Config) -> None:
    import zipfile

    global _RESUME_DONE
    if _RESUME_DONE:
        return
    _RESUME_DONE = True

    src = find_snapshot(cfg)
    if src is None:
        return

    dest = Path(cfg.out_root).parent
    dest.mkdir(parents=True, exist_ok=True)

    restored = skipped = 0
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if (dest / info.filename).exists():
                skipped += 1
                continue
            zf.extract(info, dest)
            restored += 1

    print(f"resumed from {src}  ({restored} restored, {skipped} already local)")
    for m in sorted(Path(cfg.out_root).glob("leverF__*")):
        done = (m / "untouched_test_predictions.csv").exists()
        mid = (m / "checkpoint.pt").exists()
        print(f"  {m.name:<52} {'complete' if done else 'in progress' if mid else 'empty'}")


def save_checkpoint(path: Path, model, ema, optimizer, scheduler, scaler,
                    phase: str, epoch: int) -> None:
    tmp = path.with_name(path.name + ".partial")
    torch.save({
        "model": model.state_dict(),
        "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "phase": phase,
        "epoch": epoch,
    }, tmp)
    tmp.replace(path)


def save_final_weights(path: Path, model: nn.Module, cfg: Config, extra: dict) -> None:
    tmp = path.with_name(path.name + ".partial")
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": cfg.backbone,
        "image_size": cfg.image_size,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "stage1": cfg.stage1,
        "loss": cfg.loss,
        "refit": cfg.refit_on_train_plus_calibration,
        **extra,
    }, tmp)
    tmp.replace(path)
    print(f"  saved weights -> {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def build_png_cache(cfg: Config) -> None:
    import cv2
    import pydicom

    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cfg.manifest_csv)

    done = skipped = 0
    for row in manifest.itertuples(index=False):
        dst = cache / f"{row.image_id}.png"
        if dst.exists():
            skipped += 1
            continue
        src = Path(cfg.rsna_root) / "stage_2_train_images" / f"{row.image_id}.dcm"
        arr = pydicom.dcmread(str(src)).pixel_array
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
            arr = (arr * 255).astype(np.uint8)
        arr = cv2.resize(arr, (cfg.cache_size, cfg.cache_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), arr)
        done += 1
        if done % 2000 == 0:
            print(f"  cached {done} (skipped {skipped})", flush=True)

    print(f"cache complete: {done} written, {skipped} present -> {cache}")
    pack_cache(cache, Path(cfg.archive_path))


def pack_cache(cache: Path, zip_path: Path) -> Path:
    import subprocess
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"archiving {cache} -> {zip_path}", flush=True)
    try:
        subprocess.run(["zip", "-0", "-q", "-r", str(zip_path), cache.name],
                       cwd=str(cache.parent), check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for png in sorted(cache.glob("*.png")):
                zf.write(png, f"{cache.name}/{png.name}")

    print(f"archived: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    print("=" * 70)
    print("Save a notebook version now (Quick Save), before changing anything.")
    print("/kaggle/working is wiped on restart; only a saved version keeps the cache.")
    print("=" * 70, flush=True)
    return zip_path


def build_transforms(cfg: Config, train: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    size = cfg.image_size
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if not train:
        return A.Compose([A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()])

    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
                           border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class FrameDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, cfg: Config, train: bool):
        self.rows = frame.reset_index(drop=True)
        self.cfg = cfg
        self.tf = build_transforms(cfg, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import cv2

        row = self.rows.iloc[i]
        img = cv2.imread(str(row.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {row.path}")
        img = np.stack([img] * 3, axis=-1)
        out = self.tf(image=img)
        return out["image"], torch.tensor(int(row.label), dtype=torch.long)


BACKBONE_ALIASES = {
    "efficientnet_b4": "tf_efficientnet_b4.ns_jft_in1k",
    "tf_efficientnet_b4_ns": "tf_efficientnet_b4.ns_jft_in1k",
    "densenet121": "densenet121.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
}


def create_encoder(backbone: str, pretrained: bool):
    import timm

    last_error = None
    for name in dict.fromkeys([backbone, BACKBONE_ALIASES.get(backbone, backbone)]):
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0,
                                     global_pool="", in_chans=3)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"could not build backbone {backbone!r}. If this is a weight-download failure, "
        f"enable Internet in the notebook settings. Underlying error: {last_error}")


class BinaryNet(nn.Module):

    def __init__(self, backbone: str, pretrained: bool = True, cfg: Config | None = None):
        super().__init__()
        self.encoder = create_encoder(backbone, pretrained)
        self.dropout = nn.Dropout(cfg.dropout if cfg else 0.3)
        self.classifier = nn.Linear(self.encoder.num_features, 2)

    def feature_map(self, x):
        return self.encoder(x)

    def forward(self, x):
        fmap = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return self.classifier(self.dropout(pooled))

    def head_parameters(self):
        return list(self.classifier.parameters())


class ModelEma:

    def __init__(self, model: nn.Module, decay: float):
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(v)


def compute_loss(logits: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.loss == "ce":
        return F.cross_entropy(logits, target)
    if cfg.loss == "ce_smoothing":
        return F.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
    if cfg.loss != "focal_smoothing":
        raise ValueError(f"unknown loss variant: {cfg.loss}")

    logp = F.log_softmax(logits, dim=1)
    n_cls = logits.shape[1]
    smooth = cfg.label_smoothing
    with torch.no_grad():
        dist = torch.full_like(logp, smooth / (n_cls - 1))
        dist.scatter_(1, target.unsqueeze(1), 1.0 - smooth)

    p = logp.exp()
    focal = (1.0 - p).pow(cfg.focal_gamma)
    alpha = torch.tensor([1.0 - cfg.focal_alpha_positive, cfg.focal_alpha_positive],
                         device=logits.device, dtype=logp.dtype)
    return -(alpha.unsqueeze(0) * focal * dist * logp).sum(dim=1).mean()


def load_nih_frame(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.nih_root)
    entry = next(root.glob("Data_Entry_2017*.csv"))
    df = pd.read_csv(entry)

    findings = set(cfg.nih_opacity_findings)
    df["label"] = df["Finding Labels"].apply(
        lambda s: int(bool(findings & set(str(s).split("|")))))

    index = {p.name: p for p in root.glob("**/images/*.png")}
    df["path"] = df["Image Index"].map(index)
    df = df.dropna(subset=["path"])
    if df.empty:
        raise SystemExit(f"no NIH images found under {root}")

    if cfg.nih_exclude_csv:
        excl_path = Path(cfg.nih_exclude_csv)
        if not excl_path.is_file():
            raise SystemExit(f"--nih-exclude-csv not found: {excl_path}")
        excl = set(pd.read_csv(excl_path)["nih_image_index"])
        before = len(df)
        df = df[~df["Image Index"].isin(excl)]
        print(f"NIH overlap exclusion: dropped {before - len(df):,} of {before:,} "
              f"images that pixel-match RSNA ({len(excl):,} on the list)")
        if df.empty:
            raise SystemExit("every NIH image was excluded; check the exclude list")

    rng = np.random.default_rng(cfg.seed)
    parts = []
    for lab in (0, 1):
        sub = df[df.label == lab]
        if len(sub) > cfg.nih_subset_per_class:
            sub = sub.iloc[rng.permutation(len(sub))[:cfg.nih_subset_per_class]]
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)

    if cfg.debug:
        out = out.groupby("label", group_keys=False).head(cfg.debug_stage1_per_class)
    print(f"NIH stage-1: {len(out):,} images "
          f"({int((out.label == 1).sum()):,} positive)")
    return out[["path", "label"]]


def load_kermany_frame(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.kermany_root)
    rows = []
    for split in ("train", "val", "test"):
        for cls, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
            for p in (root / "chest_xray" / split / cls).glob("*.jpeg"):
                rows.append({"path": str(p), "label": lab})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no Kermany images found under {root}")

    rng = np.random.default_rng(cfg.seed)
    normal = df[df.label == 0]
    if len(normal) > cfg.kermany_normal_limit:
        normal = normal.iloc[rng.permutation(len(normal))[:cfg.kermany_normal_limit]]
    out = pd.concat([normal, df[df.label == 1]], ignore_index=True)

    if cfg.debug:
        out = out.groupby("label", group_keys=False).head(cfg.debug_stage1_per_class)
    print(f"Kermany stage-1: {len(out):,} images "
          f"({int((out.label == 1).sum()):,} positive)")
    return out


def stage1_weights_path(cfg: Config) -> Path:
    suffix = "__clean" if cfg.nih_exclude_csv else ""
    return Path(cfg.out_root) / f"stage1__{cfg.stage1}__{cfg.backbone}{suffix}.pt"


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(int(total_steps * warmup_frac), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


PHASE_ORDER = {"head": 0, "full": 1}


def train_model(cfg: Config, train_df: pd.DataFrame, device: str, run_dir: Path,
                cal_df: pd.DataFrame | None = None,
                head_epochs: int | None = None, full_epochs: int | None = None,
                head_lr: float | None = None, backbone_lr: float | None = None,
                init_state: dict | None = None,
                fixed_epochs: dict | None = None) -> tuple[nn.Module, dict]:
    import copy
    from sklearn.metrics import roc_auc_score

    head_epochs = cfg.head_epochs if head_epochs is None else head_epochs
    full_epochs = cfg.full_epochs if full_epochs is None else full_epochs
    head_lr = cfg.head_lr if head_lr is None else head_lr
    backbone_lr = cfg.backbone_lr if backbone_lr is None else backbone_lr

    if fixed_epochs is not None:
        head_epochs = fixed_epochs.get("head", head_epochs)
        full_epochs = fixed_epochs.get("full", full_epochs)

    model = BinaryNet(cfg.backbone, pretrained=True, cfg=cfg).to(device)
    if init_state is not None:
        missing = model.load_state_dict(init_state, strict=False)
        print(f"  loaded stage-1 weights (missing keys: {len(missing.missing_keys)})")
    ema = ModelEma(model, cfg.ema_decay)

    loader = DataLoader(
        FrameDataset(train_df, cfg, train=True),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)

    scaler = torch.cuda.amp.GradScaler()
    steps_per_epoch = max(len(loader), 1)

    track_best = cfg.select_best_epoch and cal_df is not None and fixed_epochs is None
    y_cal = cal_df.label.values if track_best else None
    best = {"auc": -1.0, "state": None, "phase": None, "epoch": None}
    chosen = {"head": head_epochs, "full": full_epochs}
    history = []

    ckpt_path = run_dir / "checkpoint.pt"
    state = None
    start_phase, start_epoch = "head", 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        scaler.load_state_dict(state["scaler"])
        start_phase, start_epoch = state["phase"], state["epoch"] + 1
        print(f"  resuming: phase={start_phase}, next epoch={start_epoch + 1}", flush=True)

    for phase, n_epochs in (("head", head_epochs), ("full", full_epochs)):
        if n_epochs == 0 or PHASE_ORDER[phase] < PHASE_ORDER[start_phase]:
            continue

        if phase == "head":
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            optimizer = torch.optim.AdamW(model.head_parameters(), lr=head_lr,
                                          weight_decay=cfg.weight_decay)
        else:
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": backbone_lr},
                {"params": model.head_parameters(), "lr": head_lr * 0.1},
            ], weight_decay=cfg.weight_decay)

        scheduler = make_scheduler(optimizer, steps_per_epoch * n_epochs, cfg.warmup_frac)

        first_epoch = start_epoch if phase == start_phase else 0
        if first_epoch >= n_epochs:
            continue
        if state is not None and phase == start_phase:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])

        for epoch in range(first_epoch, n_epochs):
            import time

            t_epoch = time.time()
            model.train()
            running = 0.0
            optimizer.zero_grad(set_to_none=True)

            for img, target in loader:
                img = img.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss = compute_loss(model(img), target, cfg)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                running += loss.item()

            dt = time.time() - t_epoch
            rate = len(train_df) / max(dt, 1e-6)
            line = (f"  [{phase}] epoch {epoch + 1}/{n_epochs}  "
                    f"loss {running / steps_per_epoch:.4f}  "
                    f"{dt:.0f}s ({rate:.0f} img/s)")

            if track_best:
                p_cal = predict(ema.module, cfg, cal_df, device, views=("orig",))
                auc = float(roc_auc_score(y_cal, p_cal))
                history.append({"phase": phase, "epoch": epoch + 1,
                                "train_loss": running / steps_per_epoch, "cal_auc": auc})
                marker = ""
                if auc > best["auc"]:
                    best.update(auc=auc, phase=phase, epoch=epoch + 1,
                                state=copy.deepcopy(ema.module.state_dict()))
                    marker = "  <-- best"
                line += f"  | cal_auc {auc:.4f}{marker}"

            print(line, flush=True)
            save_checkpoint(ckpt_path, model, ema, optimizer, scheduler, scaler, phase, epoch)

            if (track_best and cfg.patience > 0 and phase == "full"
                    and best["phase"] == "full"
                    and epoch + 1 - best["epoch"] >= cfg.patience):
                print(f"  early stop: {cfg.patience} epochs without improvement "
                      f"(best was full epoch {best['epoch']})", flush=True)
                break

    if track_best and best["state"] is not None:
        pd.DataFrame(history).to_csv(run_dir / "epoch_history.csv", index=False)
        if best["phase"] == "head":
            chosen = {"head": best["epoch"], "full": 0}
        else:
            chosen = {"head": head_epochs, "full": best["epoch"]}
        print(f"  selected: {best['phase']} epoch {best['epoch']}  "
              f"cal_auc={best['auc']:.4f}", flush=True)
        ema.module.load_state_dict(best["state"])

    return ema.module, chosen


@torch.no_grad()
def predict(model: nn.Module, cfg: Config, frame: pd.DataFrame, device: str,
            views: tuple | None = None) -> np.ndarray:
    views = views if views is not None else cfg.tta
    model.eval()
    acc = []

    for view in views:
        loader = DataLoader(FrameDataset(frame, cfg, train=False),
                            batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
        probs = []
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
            if view == "hflip":
                img = torch.flip(img, dims=[3])
            with torch.cuda.amp.autocast():
                logits = model(img)
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        acc.append(torch.cat(probs))

    return torch.stack(acc).mean(0).numpy()


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1,
            "mcc": (tp * tn - fp * fn) / denom,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fit_threshold(cfg: Config, y: np.ndarray, score: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(cfg.threshold_min, cfg.threshold_max + 1e-9,
                              cfg.threshold_step), 4)
    picks = []
    for _ in range(cfg.threshold_bootstrap_iterations):
        idx = rng.integers(0, len(y), len(y))
        ys, ss = y[idx], score[idx]
        if len(np.unique(ys)) < 2:
            continue
        f1s = [binary_metrics(ys, (ss >= t).astype(int))["f1"] for t in grid]
        picks.append(grid[int(np.argmax(f1s))])
    return float(np.median(picks))


def bootstrap_ci(cfg: Config, y: np.ndarray, score: np.ndarray, thr: float,
                 seed: int) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "mcc"]
    acc: dict = {k: [] for k in keys}
    acc["auc"] = []
    for _ in range(cfg.bootstrap_iterations):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], (score[idx] >= thr).astype(int))
        for k in keys:
            acc[k].append(m[k])
        acc["auc"].append(roc_auc_score(y[idx], score[idx]))
    return pd.DataFrame([
        {"metric": k, "mean": float(np.mean(v)),
         "ci_95_lower": float(np.percentile(v, 2.5)),
         "ci_95_upper": float(np.percentile(v, 97.5))}
        for k, v in acc.items()])


def load_splits(cfg: Config):
    manifest = pd.read_csv(cfg.manifest_csv)
    manifest["path"] = manifest.image_id.apply(lambda i: str(Path(cfg.cache_dir) / f"{i}.png"))
    if cfg.debug:
        manifest = manifest.groupby(["split", "class_name"], group_keys=False).head(
            cfg.debug_per_class)
    return (manifest[manifest.split == "train"],
            manifest[manifest.split == "calibration"],
            manifest[manifest.split == "untouched_test"])


def stage_stage1(cfg: Config) -> None:
    if cfg.stage1 == "none":
        print("--stage1 none: nothing to pretrain")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    autolocate(cfg)

    dst = stage1_weights_path(cfg)
    if dst.exists() and not cfg.force_retrain:
        print(f"stage-1 weights already present: {dst.name} (use --force-retrain to redo)")
        return

    frame = load_nih_frame(cfg) if cfg.stage1 == "nih" else load_kermany_frame(cfg)
    frac = (cfg.nih_calibration_fraction if cfg.stage1 == "nih"
            else cfg.kermany_calibration_fraction)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(frame))
    n_cal = int(len(frame) * frac)
    cal = frame.iloc[perm[:n_cal]]
    trn = frame.iloc[perm[n_cal:]]

    run_dir = Path(cfg.out_root) / stage1_weights_path(cfg).stem
    run_dir.mkdir(parents=True, exist_ok=True)

    head_e = cfg.nih_head_epochs if cfg.stage1 == "nih" else cfg.kermany_head_epochs
    full_e = cfg.nih_full_epochs if cfg.stage1 == "nih" else cfg.kermany_full_epochs

    print(f"\nstage-1 ({cfg.stage1}) on {cfg.backbone}: "
          f"{len(trn):,} train / {len(cal):,} calibration")
    model, _ = train_model(cfg, trn, device, run_dir, cal_df=cal,
                           head_epochs=head_e, full_epochs=full_e,
                           head_lr=cfg.kermany_head_lr,
                           backbone_lr=cfg.kermany_backbone_lr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dst)
    print(f"stage-1 weights saved -> {dst}")
    make_snapshot(cfg)


def stage_train(cfg: Config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    autolocate(cfg)
    unpack_resume(cfg)

    train_df, cal_df, test_df = load_splits(cfg)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    if ((run_dir / "untouched_test_predictions.csv").exists()
            and not cfg.force_retrain):
        print(f"{run_dir.name} already complete - skipping")
        return

    with open(run_dir / "run_configuration.json", "w") as fh:
        json.dump(asdict(cfg), fh, indent=2, default=str)

    stop = start_snapshot_thread(cfg)

    init_state = None
    if cfg.stage1 != "none":
        wpath = stage1_weights_path(cfg)
        if not wpath.exists():
            raise SystemExit(
                f"stage-1 weights missing: {wpath}\n"
                f"Run:  --stage stage1 --stage1 {cfg.stage1} --backbone {cfg.backbone}")
        init_state = torch.load(wpath, map_location="cpu")

    print(f"\n=== {run_dir.name} ===")
    print(f"stage1={cfg.stage1}  loss={cfg.loss}  "
          f"refit={cfg.refit_on_train_plus_calibration}  tta={cfg.tta}")
    print(f"train {len(train_df):,} / calibration {len(cal_df):,} / test {len(test_df):,}")

    model, chosen = train_model(cfg, train_df, device, run_dir, cal_df=cal_df,
                                init_state=init_state)

    p_cal = predict(model, cfg, cal_df, device)
    pd.DataFrame({"image_id": cal_df.image_id.values, "label": cal_df.label.values,
                  "probability": p_cal}).to_csv(
        run_dir / "calibration_predictions.csv", index=False)

    if cfg.refit_on_train_plus_calibration:
        save_final_weights(run_dir / "model_prerefit.pt", model, cfg,
                           {"selected_epoch_counts": chosen, "trained_on": "train"})

    if cfg.refit_on_train_plus_calibration:
        print("\n  refit on train + calibration (85%) at the selected epoch counts")
        combined = pd.concat([train_df, cal_df], ignore_index=True)
        refit_dir = run_dir / "refit"
        refit_dir.mkdir(exist_ok=True)
        model, _ = train_model(cfg, combined, device, refit_dir, cal_df=None,
                               init_state=init_state, fixed_epochs=chosen)
        save_final_weights(run_dir / "model_final.pt", model, cfg,
                           {"selected_epoch_counts": chosen,
                            "trained_on": "train+calibration"})
    else:
        save_final_weights(run_dir / "model_final.pt", model, cfg,
                           {"selected_epoch_counts": chosen, "trained_on": "train"})

    p_test = predict(model, cfg, test_df, device)
    p_test_notta = predict(model, cfg, test_df, device, views=("orig",))
    pd.DataFrame({"image_id": test_df.image_id.values, "label": test_df.label.values,
                  "probability": p_test,
                  "probability_no_tta": p_test_notta}).to_csv(
        run_dir / "untouched_test_predictions.csv", index=False)

    print(f"\n  wrote predictions for {len(test_df):,} test cases")
    if stop is not None:
        stop.set()
    make_snapshot(cfg)


def stage_evaluate(cfg: Config) -> None:
    from sklearn.metrics import roc_auc_score

    autolocate(cfg)
    unpack_resume(cfg)

    members = sorted(Path(cfg.out_root).glob(f"leverF__{cfg.arm}__*"))
    members = [m for m in members if (m / "untouched_test_predictions.csv").exists()]
    if not members:
        raise SystemExit(f"no completed members for arm {cfg.arm!r} under {cfg.out_root}")

    member_cfg = json.loads((members[0] / "run_configuration.json").read_text())
    for key in ("stage1", "loss", "refit_on_train_plus_calibration", "nih_exclude_csv"):
        if key in member_cfg:
            setattr(cfg, key, member_cfg[key])

    print(f"\n=== evaluate arm {cfg.arm}: {len(members)} members ===")
    for m in members:
        print(f"  {m.name}")

    def stack(fname: str, column: str) -> pd.DataFrame:
        frames = []
        for m in members:
            df = pd.read_csv(m / fname)[["image_id", "label", column]]
            frames.append(df.rename(columns={column: m.name}))
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=["image_id", "label"], validate="one_to_one")
        return merged.sort_values("image_id").reset_index(drop=True)

    cal = stack("calibration_predictions.csv", "probability")
    test = stack("untouched_test_predictions.csv", "probability")
    test_notta = stack("untouched_test_predictions.csv", "probability_no_tta")
    names = [m.name for m in members]

    aucs = {n: float(roc_auc_score(cal.label.values, cal[n].values)) for n in names}
    total = sum(aucs.values())
    weights = {n: aucs[n] / total for n in names}

    def blend(df: pd.DataFrame) -> np.ndarray:
        return sum(weights[n] * df[n].values for n in names)

    s_cal, s_test, s_test_notta = blend(cal), blend(test), blend(test_notta)
    y_cal, y_test = cal.label.values, test.label.values

    thr = fit_threshold(cfg, y_cal, s_cal, cfg.seed)
    print(f"\nthreshold locked on calibration only: {thr:.2f}")

    out = Path(cfg.out_root) / f"arm_summary__{cfg.arm}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, score in (("with_tta", s_test), ("no_tta", s_test_notta)):
        m = binary_metrics(y_test, (score >= thr).astype(int))
        m["roc_auc"] = float(roc_auc_score(y_test, score))
        rows.append({"arm": cfg.arm, "tta": tag, "threshold": thr, "n": len(y_test), **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "test_metrics.csv", index=False)
    print("\n--- untouched test ---")
    print(summary.to_string(index=False))

    bootstrap_ci(cfg, y_test, s_test, thr, cfg.seed).to_csv(
        out / "test_bootstrap_ci.csv", index=False)

    pd.DataFrame({"image_id": test.image_id.values, "label": y_test,
                  "ensemble_probability": s_test,
                  "ensemble_probability_no_tta": s_test_notta,
                  "locked_threshold": thr}).to_csv(
        out / "untouched_test_predictions_ensemble.csv", index=False)

    with open(out / "locked_operating_point.json", "w") as fh:
        json.dump({"selected_using": "RSNA calibration split only",
                   "test_data_used_for_selection": False,
                   "arm": cfg.arm, "stage1": cfg.stage1, "loss": cfg.loss,
                   "nih_exclude_csv": cfg.nih_exclude_csv,
                   "refit_on_train_plus_calibration": cfg.refit_on_train_plus_calibration,
                   "members": names, "member_calibration_auc": aucs,
                   "ensemble_weights": weights, "decision_threshold": thr}, fh, indent=2)

    print(f"\nwritten to {out}")
    print("next: leverF_matched_evaluation.py reads these files for the paper's tables")
    make_snapshot(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "stage1", "train", "evaluate"])
    ap.add_argument("--arm", default=Config.arm,
                    help="run-group name, e.g. stage1none / lossce / norefit")
    ap.add_argument("--backbone", default=Config.backbone, choices=list(BACKBONE_ALIASES))
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--stage1", default=Config.stage1, choices=["nih", "kermany", "none"])
    ap.add_argument("--loss", default=Config.loss,
                    choices=["focal_smoothing", "ce_smoothing", "ce"])
    ap.add_argument("--no-refit", action="store_true",
                    help="don't refit on train + calibration")
    ap.add_argument("--no-tta", action="store_true",
                    help="score the original view only (both are saved anyway)")
    ap.add_argument("--image-size", type=int, default=Config.image_size)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--head-epochs", type=int, default=Config.head_epochs)
    ap.add_argument("--full-epochs", type=int, default=Config.full_epochs)
    ap.add_argument("--patience", type=int, default=Config.patience,
                    help="stop after N full epochs without a better calibration AUC (0 = off)")
    ap.add_argument("--manifest-csv", default=Config.manifest_csv)
    ap.add_argument("--rsna-root", default=Config.rsna_root)
    ap.add_argument("--nih-root", default=Config.nih_root)
    ap.add_argument("--nih-exclude-csv", default=Config.nih_exclude_csv,
                    help="nih_exclude_list.csv from nih_rsna_overlap_audit.py")
    ap.add_argument("--kermany-root", default=Config.kermany_root)
    ap.add_argument("--cache-dir", default=Config.cache_dir)
    ap.add_argument("--out-root", default=Config.out_root)
    ap.add_argument("--num-workers", type=int, default=Config.num_workers)
    ap.add_argument("--snapshot-every-min", type=int, default=Config.snapshot_every_min)
    ap.add_argument("--snapshot-path", default=Config.snapshot_path)
    ap.add_argument("--resume-from", default=Config.resume_from)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="tiny subset, just to check everything runs")
    args = ap.parse_args()

    cfg = Config(
        arm=args.arm, backbone=args.backbone, seed=args.seed, stage1=args.stage1,
        loss=args.loss,
        refit_on_train_plus_calibration=not args.no_refit,
        tta=("orig",) if args.no_tta else ("orig", "hflip"),
        image_size=args.image_size, batch_size=args.batch_size,
        head_epochs=args.head_epochs, full_epochs=args.full_epochs,
        patience=args.patience,
        manifest_csv=args.manifest_csv, rsna_root=args.rsna_root,
        nih_root=args.nih_root, nih_exclude_csv=args.nih_exclude_csv,
        kermany_root=args.kermany_root,
        cache_dir=args.cache_dir, out_root=args.out_root,
        num_workers=args.num_workers, snapshot_every_min=args.snapshot_every_min,
        snapshot_path=args.snapshot_path, resume_from=args.resume_from,
        force_retrain=args.force_retrain, debug=args.debug,
    )

    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    if cfg.debug:
        print("DEBUG run on a tiny subset - the numbers mean nothing")

    if args.stage == "cache":
        autolocate(cfg)
        build_png_cache(cfg)
    elif args.stage == "stage1":
        stage_stage1(cfg)
    elif args.stage == "train":
        stage_train(cfg)
    else:
        stage_evaluate(cfg)


if __name__ == "__main__":
    main()
