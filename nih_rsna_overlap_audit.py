from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


HASH_BITS = 64
CHUNK_BITS = 16
DEFAULT_MAX_DISTANCE = 3


def dhash(img: np.ndarray) -> np.uint64:
    import cv2

    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    out = np.uint64(0)
    for i, b in enumerate(bits):
        if b:
            out |= np.uint64(1) << np.uint64(i)
    return out


def hamming(a: np.ndarray, b: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(a, b)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def locate(patterns: list[str], what: str) -> Path:
    for base in ("/kaggle/input", "/kaggle/working"):
        root = Path(base)
        if not root.is_dir():
            continue
        for pattern in patterns:
            hits = sorted(root.glob(pattern))
            if hits:
                return hits[0]
    raise SystemExit(f"could not locate {what}. Attach it as a notebook input.")


def hash_rsna(manifest: pd.DataFrame, rsna_root: Path, cache_dir: Path | None) -> dict:
    import cv2

    use_cache = cache_dir is not None and cache_dir.is_dir() and any(cache_dir.glob("*.png"))
    print(f"RSNA: hashing {len(manifest):,} images "
          f"({'PNG cache' if use_cache else 'DICOM decode'})", flush=True)

    if not use_cache:
        import pydicom

    out: dict = {}
    t0 = time.time()
    for n, row in enumerate(manifest.itertuples(index=False), 1):
        if use_cache:
            img = cv2.imread(str(cache_dir / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
        else:
            path = rsna_root / "stage_2_train_images" / f"{row.image_id}.dcm"
            if not path.exists():
                continue
            img = pydicom.dcmread(str(path)).pixel_array
        out[row.image_id] = dhash(img)
        if n % 5000 == 0:
            print(f"  {n:,} / {len(manifest):,}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"RSNA: {len(out):,} hashed in {time.time() - t0:.0f}s")
    return out


def hash_nih(nih_root: Path) -> dict:
    import cv2

    files = sorted(nih_root.glob("**/images/*.png"))
    if not files:
        raise SystemExit(f"no NIH images found under {nih_root}")
    print(f"NIH: hashing {len(files):,} images", flush=True)

    out: dict = {}
    t0 = time.time()
    for n, path in enumerate(files, 1):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[path.name] = dhash(img)
        if n % 10000 == 0:
            print(f"  {n:,} / {len(files):,}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"NIH: {len(out):,} hashed in {time.time() - t0:.0f}s")
    return out


def build_index(hashes: np.ndarray) -> list[dict]:
    index = []
    for c in range(HASH_BITS // CHUNK_BITS):
        shift = np.uint64(c * CHUNK_BITS)
        mask = np.uint64((1 << CHUNK_BITS) - 1)
        keys = (hashes >> shift) & mask
        bucket: dict = {}
        for i, k in enumerate(keys):
            bucket.setdefault(int(k), []).append(i)
        index.append(bucket)
    return index


def find_matches(rsna: dict, nih: dict, max_distance: int) -> pd.DataFrame:
    nih_names = np.array(list(nih.keys()))
    nih_hashes = np.array(list(nih.values()), dtype=np.uint64)
    index = build_index(nih_hashes)

    print(f"matching {len(rsna):,} RSNA against {len(nih_hashes):,} NIH "
          f"(Hamming <= {max_distance})", flush=True)

    rows = []
    t0 = time.time()
    for n, (image_id, h) in enumerate(rsna.items(), 1):
        candidates: set = set()
        for c, bucket in enumerate(index):
            shift = np.uint64(c * CHUNK_BITS)
            mask = np.uint64((1 << CHUNK_BITS) - 1)
            key = int((np.uint64(h) >> shift) & mask)
            hit = bucket.get(key)
            if hit:
                candidates.update(hit)
        if not candidates:
            continue

        idx = np.fromiter(candidates, dtype=np.int64)
        dist = hamming(nih_hashes[idx], np.uint64(h))
        keep = dist <= max_distance
        if not keep.any():
            continue

        best = int(np.argmin(dist))
        rows.append({"image_id": image_id,
                     "nih_image_index": str(nih_names[idx[best]]),
                     "hamming": int(dist[best]),
                     "n_nih_candidates": int(keep.sum())})

        if n % 5000 == 0:
            print(f"  {n:,} / {len(rsna):,}  ({time.time() - t0:.0f}s, "
                  f"{len(rows):,} matched)", flush=True)

    print(f"matched {len(rows):,} RSNA images in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def stage1_sample(nih_root: Path, seed: int, per_class: int,
                  findings: tuple) -> set:
    entry = next(nih_root.glob("Data_Entry_2017*.csv"))
    df = pd.read_csv(entry)
    fset = set(findings)
    df["label"] = df["Finding Labels"].apply(
        lambda s: int(bool(fset & set(str(s).split("|")))))

    available = {p.name for p in nih_root.glob("**/images/*.png")}
    df = df[df["Image Index"].isin(available)]

    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        sub = df[df.label == lab]
        if len(sub) > per_class:
            sub = sub.iloc[rng.permutation(len(sub))[:per_class]]
        parts.append(sub)
    return set(pd.concat(parts)["Image Index"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--rsna-root", default="")
    ap.add_argument("--nih-root", default="")
    ap.add_argument("--cache-dir", default="/tmp/png_cache")
    ap.add_argument("--out-dir", default="overlap_audit")
    ap.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE,
                    help="largest Hamming distance counted as the same radiograph")
    ap.add_argument("--seed", type=int, default=42,
                    help="same seed as the stage-1 run")
    ap.add_argument("--nih-per-class", type=int, default=25000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = (Path(args.manifest_csv) if args.manifest_csv
                     else locate(["**/rsna_split_manifest.csv"], "rsna_split_manifest.csv"))
    rsna_root = (Path(args.rsna_root) if args.rsna_root
                 else locate(["**/rsna-pneumonia-detection-challenge"], "the RSNA competition data"))
    nih_root = (Path(args.nih_root) if args.nih_root
                else locate(["**/Data_Entry_2017*.csv"], "NIH CXR-14").parent)

    print(f"manifest : {manifest_path}")
    print(f"RSNA     : {rsna_root}")
    print(f"NIH      : {nih_root}\n")

    manifest = pd.read_csv(manifest_path)

    rsna = hash_rsna(manifest, rsna_root, Path(args.cache_dir))
    nih = hash_nih(nih_root)
    pairs = find_matches(rsna, nih, args.max_distance)

    split_of = dict(zip(manifest.image_id, manifest.split))
    if len(pairs):
        pairs["split"] = pairs.image_id.map(split_of)
    else:
        pairs = pd.DataFrame(columns=["image_id", "nih_image_index", "hamming",
                                      "n_nih_candidates", "split"])
    pairs.to_csv(out / "overlap_pairs.csv", index=False)

    sampled = stage1_sample(nih_root, args.seed, args.nih_per_class,
                            ("Infiltration", "Consolidation", "Pneumonia"))
    pairs["in_stage1_sample"] = pairs.nih_image_index.isin(sampled)

    counts = manifest.split.value_counts().to_dict()
    summary = {
        "rsna_images_hashed": len(rsna),
        "nih_images_hashed": len(nih),
        "max_hamming_distance": args.max_distance,
        "stage1_sample_size": len(sampled),
        "stage1_seed": args.seed,
        "rsna_matched_to_nih_total": int(len(pairs)),
        "by_split": {},
    }
    for split in ("train", "calibration", "untouched_test"):
        sub = pairs[pairs.split == split]
        total = int(counts.get(split, 0))
        seen = int(sub.in_stage1_sample.sum())
        summary["by_split"][split] = {
            "images_in_split": total,
            "matched_to_nih": int(len(sub)),
            "matched_pct": round(100 * len(sub) / total, 2) if total else 0.0,
            "seen_during_stage1": seen,
            "seen_pct": round(100 * seen / total, 2) if total else 0.0,
        }

    with open(out / "overlap_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    excl = sorted(set(pairs.nih_image_index))
    pd.DataFrame({"nih_image_index": excl}).to_csv(
        out / "nih_exclude_list.csv", index=False)

    print("\n" + "=" * 74)
    print("NIH / RSNA overlap")
    print("=" * 74)
    print(f"RSNA images matched to an NIH image: {len(pairs):,} / {len(rsna):,} "
          f"({100 * len(pairs) / max(len(rsna), 1):.1f}%)")
    print(f"Stage-1 sampled {len(sampled):,} NIH images (seed {args.seed})\n")
    print(f"{'split':<16}{'images':>9}{'matched':>10}{'%':>8}"
          f"{'seen in stage-1':>18}{'%':>8}")
    for split, s in summary["by_split"].items():
        print(f"{split:<16}{s['images_in_split']:>9,}{s['matched_to_nih']:>10,}"
              f"{s['matched_pct']:>8.1f}{s['seen_during_stage1']:>18,}{s['seen_pct']:>8.1f}")

    test = summary["by_split"]["untouched_test"]
    print("\n" + "-" * 74)
    print("Test images seen during stage-1")
    print("-" * 74)
    print(f"{test['seen_during_stage1']:,} of {test['images_in_split']:,} untouched-test "
          f"images ({test['seen_pct']:.1f}%) were shown to the model during stage-1.")
    if test["seen_during_stage1"] == 0:
        print("\nNo overlap with the stage-1 sample.")
    else:
        print("\nSome test images were in the stage-1 sample. Options: rerun stage-1 with\n"
              "--nih-exclude-csv nih_exclude_list.csv, or use Kermany / no stage-1.\n"
              "(Only the images overlap; stage-1 used NIH's own labels.)")
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
