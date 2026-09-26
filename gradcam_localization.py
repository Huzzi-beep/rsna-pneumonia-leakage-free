from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import leverF_pipeline as lp


def grad_cam(model, image: torch.Tensor, target_class: int = 1) -> np.ndarray:
    model.eval()
    image = image.unsqueeze(0)

    fmap = model.feature_map(image)
    fmap.retain_grad()
    pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
    logits = model.classifier(pooled)

    model.zero_grad(set_to_none=True)
    logits[0, target_class].backward()

    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1)).squeeze(0)
    cam = cam.detach().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return cam


def upsample_cam(cam: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)


def boxes_to_mask(boxes: list, size: int, source_size: int = 1024) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    scale = size / source_size
    for (x, y, w, h) in boxes:
        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = int((x + w) * scale), int((y + h) * scale)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return mask


def localisation_scores(cam: np.ndarray, box_mask: np.ndarray,
                        percentile: float) -> dict:
    if not box_mask.any():
        return {}

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    hit = bool(box_mask[peak])

    thr = np.percentile(cam, percentile)
    cam_mask = cam >= thr
    inter = int((cam_mask & box_mask).sum())
    union = int((cam_mask | box_mask).sum())

    return {
        "pointing_hit": int(hit),
        "iou": inter / union if union else 0.0,
        "box_coverage": inter / int(box_mask.sum()),
        "cam_area_frac": float(cam_mask.mean()),
    }


def load_members(results_root: Path, arm: str) -> list[Path]:
    members = sorted(results_root.glob(f"leverF__{arm}__*"))
    members = [m for m in members if (m / "model_final.pt").exists()]
    if not members:
        raise SystemExit(
            f"no checkpoints for arm {arm!r} under {results_root}.\n"
            f"leverF_pipeline.py writes model_final.pt in each member directory; the "
            f"original leverD runs did not save weights at all, which is why this "
            f"analysis needs the retrained arm.")
    return members


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rsna-root",
                    default="/kaggle/input/competitions/rsna-pneumonia-detection-challenge")
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--cache-dir", default="/tmp/png_cache")
    ap.add_argument("--out-dir", default="gradcam_output")
    ap.add_argument("--cam-percentile", type=float, default=80.0,
                    help="the map is thresholded at this percentile of its own values")
    ap.add_argument("--max-cases", type=int, default=300,
                    help="number of true positives to score")
    ap.add_argument("--save-examples", type=int, default=0,
                    help="write this many overlay figures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root = Path(args.results_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = lp.Config(manifest_csv=args.manifest_csv, cache_dir=args.cache_dir,
                    rsna_root=args.rsna_root, out_root=str(results_root))
    lp.autolocate(cfg)

    _, _, test_df = lp.load_splits(cfg)
    arm_summary = results_root / f"arm_summary__{args.arm}"
    pred_path = arm_summary / "untouched_test_predictions_ensemble.csv"
    if not pred_path.exists():
        raise SystemExit(f"missing {pred_path}. Run --stage evaluate for this arm first.")

    preds = pd.read_csv(pred_path)
    thr = float(preds.locked_threshold.iloc[0])
    tp_ids = preds.loc[(preds.label == 1)
                       & (preds.ensemble_probability >= thr), "image_id"].tolist()

    boxes = pd.read_csv(Path(cfg.rsna_root) / "stage_2_train_labels.csv")
    boxes = boxes[boxes.Target == 1].dropna(subset=["x", "y", "width", "height"])
    box_index: dict = {}
    for row in boxes.itertuples(index=False):
        box_index.setdefault(row.patientId, []).append(
            (row.x, row.y, row.width, row.height))

    tp_ids = [i for i in tp_ids if i in box_index]
    rng = np.random.default_rng(args.seed)
    if len(tp_ids) > args.max_cases:
        tp_ids = [tp_ids[i] for i in rng.permutation(len(tp_ids))[:args.max_cases]]

    print(f"scoring {len(tp_ids)} true-positive cases with ground-truth boxes")

    frame = test_df[test_df.image_id.isin(set(tp_ids))].reset_index(drop=True)
    dataset = lp.FrameDataset(frame, cfg, train=False)

    members = load_members(results_root, args.arm)
    print(f"members: {[m.name for m in members]}")

    per_case_rows = []
    cam_store: dict = {}

    for member in members:
        ckpt = torch.load(member / "model_final.pt", map_location=device)
        model = lp.BinaryNet(ckpt["backbone"], pretrained=False, cfg=cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n  {member.name}  ({ckpt['backbone']})")

        for i in range(len(dataset)):
            image, _ = dataset[i]
            image_id = frame.image_id.iloc[i]
            cam = upsample_cam(grad_cam(model, image.to(device)), cfg.image_size)
            cam_store.setdefault(image_id, []).append(cam)

            box_mask = boxes_to_mask(box_index[image_id], cfg.image_size)
            scores = localisation_scores(cam, box_mask, args.cam_percentile)
            if scores:
                per_case_rows.append({"member": member.name,
                                      "backbone": ckpt["backbone"],
                                      "seed": ckpt["seed"],
                                      "image_id": image_id, **scores})

    per_case = pd.DataFrame(per_case_rows)
    per_case.to_csv(out / "per_case_localization.csv", index=False)

    ens_rows = []
    for image_id, cams in cam_store.items():
        cam = np.mean(cams, axis=0)
        box_mask = boxes_to_mask(box_index[image_id], cfg.image_size)
        scores = localisation_scores(cam, box_mask, args.cam_percentile)
        if scores:
            ens_rows.append({"member": "ensemble_mean", "backbone": "ensemble",
                             "seed": -1, "image_id": image_id, **scores})
    ensemble = pd.DataFrame(ens_rows)
    ensemble.to_csv(out / "per_case_localization_ensemble.csv", index=False)

    ids = list(cam_store.keys())
    shuffled = [ids[i] for i in rng.permutation(len(ids))]
    chance_rows = []
    for image_id, other_id in zip(ids, shuffled):
        if image_id == other_id:
            continue
        cam = np.mean(cam_store[image_id], axis=0)
        box_mask = boxes_to_mask(box_index[other_id], cfg.image_size)
        scores = localisation_scores(cam, box_mask, args.cam_percentile)
        if scores:
            chance_rows.append(scores)
    chance = pd.DataFrame(chance_rows)

    def summarise(df: pd.DataFrame, label: str) -> dict:
        n = len(df)
        pg = df.pointing_hit.mean()
        z = 1.96
        denom = 1 + z ** 2 / n
        centre = (pg + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(pg * (1 - pg) / n + z ** 2 / (4 * n ** 2)) / denom
        return {"group": label, "n": n,
                "pointing_game": round(float(pg), 4),
                "pointing_ci_lower": round(float(centre - half), 4),
                "pointing_ci_upper": round(float(centre + half), 4),
                "mean_iou": round(float(df.iou.mean()), 4),
                "median_iou": round(float(df.iou.median()), 4),
                "mean_box_coverage": round(float(df.box_coverage.mean()), 4),
                "mean_cam_area_frac": round(float(df.cam_area_frac.mean()), 4)}

    summary = [summarise(g, f"{bb}") for bb, g in per_case.groupby("backbone")]
    summary.append(summarise(ensemble, "ensemble_mean"))
    if len(chance):
        summary.append(summarise(chance, "chance (shuffled boxes)"))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "localization_summary.csv", index=False)

    print("\n--- Grad-CAM localisation on true positives ---")
    print(summary_df.to_string(index=False))

    with open(out / "settings.json", "w") as fh:
        json.dump({"arm": args.arm, "cam_percentile": args.cam_percentile,
                   "n_cases": len(frame), "image_size": cfg.image_size,
                   "locked_threshold": thr, "seed": args.seed,
                   "members": [m.name for m in members]}, fh, indent=2)

    if args.save_examples > 0:
        save_overlays(frame, cam_store, box_index, cfg, out, args.save_examples,
                      args.cam_percentile)

    print(f"\nwritten to {out.resolve()}")
    print("Compare the backbone and ensemble rows with the chance row.")


def save_overlays(frame, cam_store, box_index, cfg, out: Path, n: int,
                  percentile: float) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig_dir = out / "examples"
    fig_dir.mkdir(exist_ok=True)
    scale = cfg.image_size / 1024.0

    for i in range(min(n, len(frame))):
        image_id = frame.image_id.iloc[i]
        if image_id not in cam_store:
            continue
        img = cv2.imread(str(Path(cfg.cache_dir) / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (cfg.image_size, cfg.image_size))
        cam = np.mean(cam_store[image_id], axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("chest radiograph")
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(cam, cmap="jet", alpha=0.4)
        ax[1].contour(cam >= np.percentile(cam, percentile), levels=[0.5],
                      colors="white", linewidths=1.0)
        ax[1].set_title("Grad-CAM (ensemble mean)")

        for (x, y, w, h) in box_index.get(image_id, []):
            for a in ax:
                a.add_patch(patches.Rectangle((x * scale, y * scale), w * scale, h * scale,
                                              fill=False, edgecolor="lime", linewidth=1.5))
        for a in ax:
            a.axis("off")

        fig.suptitle(f"{image_id}  (true positive, ground-truth boxes in green)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{image_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  overlays -> {fig_dir}")


if __name__ == "__main__":
    main()
