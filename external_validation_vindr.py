from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import leverF_pipeline as lp

OPACITY = {"Lung Opacity", "Consolidation", "Infiltration"}
NO_FINDING = "No finding"


def locate_vindr(given: str) -> tuple[Path, Path]:
    roots = [Path(given)] if given else []
    roots += [Path(p) for p in ("/kaggle/input", "/kaggle/working") if Path(p).is_dir()]

    csv = None
    for r in roots:
        for c in sorted(r.glob("**/train.csv")):
            try:
                cols = set(pd.read_csv(c, nrows=1).columns)
            except Exception:
                continue
            if {"image_id", "class_name", "rad_id"} <= cols:
                csv = c
                break
        if csv:
            break
    if csv is None:
        raise SystemExit("VinDr train.csv (image_id, class_name, rad_id, ...) not found. "
                         "Attach the VinBigData resized-PNG dataset and pass --vindr-root.")

    img_dir = None
    for r in roots:
        for d in sorted(r.glob("**/train")):
            if d.is_dir() and any(d.glob("*.png")):
                img_dir = d
                break
        if img_dir:
            break
    if img_dir is None:
        raise SystemExit("no train/*.png directory found - use a resized-PNG VinDr dataset, "
                         "not the raw DICOM competition data.")
    return csv, img_dir


def image_labels(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    g = df.groupby("image_id")
    pos = g.class_name.apply(lambda s: bool(set(s) & OPACITY))
    clean = g.class_name.apply(lambda s: set(s) == {NO_FINDING})
    out = pd.DataFrame({"positive": pos, "clean_negative": clean})
    out["other"] = ~out.positive & ~out.clean_negative
    return out.reset_index()


def build_frame(labels: pd.DataFrame, img_dir: Path, definition: str,
                cap: int, seed: int) -> pd.DataFrame:
    pos = labels[labels.positive]
    if definition == "opacity_vs_nofinding":
        neg = labels[labels.clean_negative]
    elif definition == "opacity_vs_all_neg":
        neg = labels[~labels.positive]
    else:
        raise ValueError(definition)

    rng = np.random.default_rng(seed)
    if cap > 0 and len(neg) > cap:
        neg = neg.iloc[rng.permutation(len(neg))[:cap]]

    rows = []
    for _, r in pos.iterrows():
        rows.append({"image_id": r.image_id, "label": 1, "path": str(img_dir / f"{r.image_id}.png")})
    for _, r in neg.iterrows():
        rows.append({"image_id": r.image_id, "label": 0, "path": str(img_dir / f"{r.image_id}.png")})
    frame = pd.DataFrame(rows)
    missing = [p for p in frame.path if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images in train.csv have no PNG under {img_dir} "
                         f"(first: {missing[0]})")
    return frame


def evaluate(y: np.ndarray, score: np.ndarray, thr: float, label: str) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = lp.binary_metrics(y, (score >= thr).astype(int))
    return {"operating_point": label, "threshold": round(thr, 4),
            "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4), "specificity": round(m["specificity"], 4),
            "f1": round(m["f1"], 4), "mcc": round(m["mcc"], 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "auprc": round(float(average_precision_score(y, score)), 4),
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--vindr-root", default="")
    ap.add_argument("--out-dir", default="external_validation_vindr_output")
    ap.add_argument("--negative-cap", type=int, default=3000,
                    help="cap on negatives per definition; 0 uses all")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root, out = Path(a.results_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv, img_dir = locate_vindr(a.vindr_root)
    print(f"VinDr-CXR labels : {csv}")
    print(f"VinDr-CXR images : {img_dir}")

    labels = image_labels(csv)
    print(f"\nimages {len(labels):,}  positive {int(labels.positive.sum()):,}  "
          f"clean negative {int(labels.clean_negative.sum()):,}  other {int(labels.other.sum()):,}")

    op_path = results_root / f"arm_summary__{a.arm}" / "locked_operating_point.json"
    if not op_path.exists():
        raise SystemExit(f"missing {op_path} - run --stage evaluate for this arm first")
    operating = json.loads(op_path.read_text())
    rsna_thr = float(operating["decision_threshold"])
    weights = operating["ensemble_weights"]
    members = [m for m in sorted(results_root.glob(f"leverF__{a.arm}__*"))
               if (m / "model_final.pt").exists()]
    if not members:
        raise SystemExit(f"no model_final.pt for arm {a.arm!r} under {results_root}")
    print(f"RSNA locked threshold {rsna_thr:.2f}; {len(members)} members")

    cfg = lp.Config(out_root=str(results_root), batch_size=a.batch_size,
                    num_workers=a.num_workers)

    universe = build_frame(labels, img_dir, "opacity_vs_all_neg", cap=0, seed=a.seed)
    blended = np.zeros(len(universe))
    for member in members:
        ckpt = torch.load(member / "model_final.pt", map_location=device)
        model = lp.BinaryNet(ckpt["backbone"], pretrained=False, cfg=cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        blended += weights.get(member.name, 1.0 / len(members)) * lp.predict(model, cfg, universe, device)
        print(f"  scored {member.name}")
    universe["probability"] = blended
    universe.to_csv(out / "predictions_all_scored.csv", index=False)

    rows = []
    for definition in ("opacity_vs_nofinding", "opacity_vs_all_neg"):
        frame = build_frame(labels, img_dir, definition, a.negative_cap, a.seed)
        sc = frame.merge(universe[["image_id", "probability"]], on="image_id", validate="one_to_one")
        y, s = sc.label.values, sc.probability.values
        prev = float(y.mean())
        print(f"\n=== {definition} ===  n={len(y):,}  prevalence {100*prev:.1f}%")
        refit = lp.fit_threshold(cfg, y, s, a.seed)
        for label, thr in (("transferred (RSNA threshold)", rsna_thr),
                           ("re-fitted on VinDr", refit)):
            r = evaluate(y, s, thr, label)
            r.update({"definition": definition, "n": len(y), "prevalence": round(prev, 4)})
            rows.append(r)
        print(pd.DataFrame([r for r in rows if r["definition"] == definition])[
            ["operating_point", "threshold", "roc_auc", "auprc", "f1", "recall",
             "precision", "specificity"]].to_string(index=False))

    summary = pd.DataFrame(rows)[["definition", "operating_point", "n", "prevalence",
                                  "threshold", "roc_auc", "auprc", "accuracy", "precision",
                                  "recall", "specificity", "f1", "mcc", "tn", "fp", "fn", "tp"]]
    summary.to_csv(out / "external_validation_summary.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "dataset": "VinDr-CXR", "labels_csv": str(csv), "images": str(img_dir),
        "arm": a.arm, "rsna_locked_threshold": rsna_thr, "ensemble_weights": weights,
        "positive_findings": sorted(OPACITY), "negative_cap": a.negative_cap,
        "seed": a.seed, "members": [m.name for m in members]}, indent=2))

    print("\n--- external validation summary ---")
    print(summary.to_string(index=False))
    print(f"\nwritten to {out.resolve()}")
    print("\nAUC is the main number; the transferred and re-fitted threshold rows are "
          "both in the summary.")


if __name__ == "__main__":
    main()
