from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import leverF_pipeline as lp

OPACITY_COLS = ("Lung Opacity", "Consolidation", "Pneumonia")


def locate_chexpert(given: str) -> tuple[Path, Path]:
    roots = [Path(given)] if given else []
    roots += [Path(p) for p in ("/kaggle/input", "/kaggle/working") if Path(p).is_dir()]
    for r in roots:
        for c in sorted(r.glob("**/train.csv")):
            try:
                cols = set(pd.read_csv(c, nrows=1).columns)
            except Exception:
                continue
            if {"Path", "Frontal/Lateral", "Pneumonia", "No Finding"} <= cols:
                first = pd.read_csv(c, nrows=1).Path.iloc[0]
                top = first.split("/")[0]
                base = c.parent
                while base != base.parent and base.name != top:
                    base = base.parent
                return c, base.parent if base.name == top else c.parent
    raise SystemExit("CheXpert train.csv not found - attach the CheXpert-v1.0-small dataset.")


def image_labels(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    lab = df[list(OPACITY_COLS) + ["No Finding"]]
    uncertain = (lab == -1.0).any(axis=1)
    df = df[~uncertain].copy()
    df["pneumonia"] = df["Pneumonia"] == 1.0
    df["opacity"] = (df[list(OPACITY_COLS)] == 1.0).any(axis=1)
    df["clean_negative"] = df["No Finding"] == 1.0
    return df[["Path", "pneumonia", "opacity", "clean_negative"]].reset_index(drop=True)


def build_frame(labels: pd.DataFrame, root: Path, definition: str,
                cap_pos: int, cap_neg: int, seed: int) -> pd.DataFrame:
    if definition == "pneumonia_vs_nofinding":
        pos, neg = labels[labels.pneumonia], labels[labels.clean_negative]
    elif definition == "opacity_vs_nofinding":
        pos, neg = labels[labels.opacity], labels[labels.clean_negative]
    elif definition == "opacity_vs_all_neg":
        pos, neg = labels[labels.opacity], labels[~labels.opacity]
    else:
        raise ValueError(definition)
    rng = np.random.default_rng(seed)
    if cap_pos > 0 and len(pos) > cap_pos:
        pos = pos.iloc[rng.permutation(len(pos))[:cap_pos]]
    if cap_neg > 0 and len(neg) > cap_neg:
        neg = neg.iloc[rng.permutation(len(neg))[:cap_neg]]
    rows = [{"image_id": p, "label": 1, "path": str(root / p)} for p in pos.Path] + \
           [{"image_id": p, "label": 0, "path": str(root / p)} for p in neg.Path]
    frame = pd.DataFrame(rows)
    missing = [p for p in frame.path[:50] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"images not found, e.g. {missing[0]} - check --chexpert-root")
    return frame


def evaluate(y, score, thr, label):
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
    ap.add_argument("--chexpert-root", default="")
    ap.add_argument("--out-dir", default="external_validation_chexpert_output")
    ap.add_argument("--cap-positives", type=int, default=3000)
    ap.add_argument("--cap-negatives", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root, out = Path(a.results_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv, root = locate_chexpert(a.chexpert_root)
    labels = image_labels(csv)

    parts = labels.Path.iloc[0].split("/")
    strip = None
    for k in range(0, len(parts) - 2):
        if (root / "/".join(parts[k:])).exists():
            strip = k
            break
    if strip is None:
        hits = [h for h in root.rglob(parts[-1])
                if h.parent.name == parts[-2] and h.parent.parent.name == parts[-3]]
        if not hits:
            raise SystemExit(f"cannot find {labels.Path.iloc[0]} under {root}")
        h = hits[0].as_posix()
        for k in range(len(parts)):
            tail = "/".join(parts[k:])
            if h.endswith(tail):
                strip, root = k, Path(h[: -len(tail)].rstrip("/"))
                break
    if strip:
        labels = labels.assign(Path=labels.Path.str.split("/").str[strip:].str.join("/"))
    print(f"CheXpert labels: {csv}")
    print(f"CheXpert images: {root}   (stripped {strip} leading path component(s) from the CSV)")
    print(f"\nfrontal, non-uncertain images {len(labels):,}  pneumonia {int(labels.pneumonia.sum()):,}  "
          f"opacity-type {int(labels.opacity.sum()):,}  no-finding {int(labels.clean_negative.sum()):,}")

    op = json.loads((results_root / f"arm_summary__{a.arm}" / "locked_operating_point.json").read_text())
    rsna_thr, weights = float(op["decision_threshold"]), op["ensemble_weights"]
    members = [m for m in sorted(results_root.glob(f"leverF__{a.arm}__*")) if (m / "model_final.pt").exists()]
    print(f"RSNA locked threshold {rsna_thr:.2f}; {len(members)} members")
    cfg = lp.Config(out_root=str(results_root), batch_size=a.batch_size, num_workers=a.num_workers)

    frames = {d: build_frame(labels, root, d, a.cap_positives, a.cap_negatives, a.seed)
              for d in ("pneumonia_vs_nofinding", "opacity_vs_nofinding", "opacity_vs_all_neg")}
    universe = pd.concat(frames.values()).drop_duplicates("image_id").reset_index(drop=True)
    print(f"scoring {len(universe):,} unique images")
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
    for definition, frame in frames.items():
        sc = frame.merge(universe[["image_id", "probability"]], on="image_id", validate="one_to_one")
        y, s = sc.label.values, sc.probability.values
        prev = float(y.mean())
        print(f"\n=== {definition} ===  n={len(y):,}  prevalence {100*prev:.1f}%")
        refit = lp.fit_threshold(cfg, y, s, a.seed)
        for label, thr in (("transferred (RSNA threshold)", rsna_thr), ("re-fitted on CheXpert", refit)):
            r = evaluate(y, s, thr, label)
            r.update({"definition": definition, "n": len(y), "prevalence": round(prev, 4)})
            rows.append(r)
        print(pd.DataFrame([r for r in rows if r["definition"] == definition])[
            ["operating_point", "threshold", "roc_auc", "auprc", "f1", "recall",
             "precision", "specificity"]].to_string(index=False))

    summary = pd.DataFrame(rows)[["definition", "operating_point", "n", "prevalence", "threshold",
                                  "roc_auc", "auprc", "accuracy", "precision", "recall",
                                  "specificity", "f1", "mcc", "tn", "fp", "fn", "tp"]]
    summary.to_csv(out / "external_validation_summary.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "dataset": "CheXpert-v1.0-small (frontal, uncertain excluded)", "labels_csv": str(csv),
        "arm": a.arm, "rsna_locked_threshold": rsna_thr, "ensemble_weights": weights,
        "opacity_columns": OPACITY_COLS, "cap_positives": a.cap_positives,
        "cap_negatives": a.cap_negatives, "seed": a.seed,
        "members": [m.name for m in members]}, indent=2))
    print("\n--- external validation summary ---")
    print(summary.to_string(index=False))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
