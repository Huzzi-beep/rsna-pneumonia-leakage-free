from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OPACITY = {"Lung Opacity", "Consolidation", "Infiltration"}
NO_FINDING = "No finding"


def roc_auc(y, s):
    r = stats.rankdata(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def bootstrap_auc(y, s, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def reader_counts(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df["opacity"] = df.class_name.isin(OPACITY)
    df["nofind"] = df.class_name == NO_FINDING
    g = df.groupby("image_id")
    out = pd.DataFrame({
        "readers": g.rad_id.nunique(),
        "opacity_readers": g.apply(lambda d: d.loc[d.opacity, "rad_id"].nunique()),
        "nofind_readers": g.apply(lambda d: d.loc[d.nofind, "rad_id"].nunique()),
    })
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--resnet50", default="", help="optional: baseline predictions for the same images")
    ap.add_argument("--out-dir", default="", help="where to write the two tables (default: next to --predictions)")
    a = ap.parse_args()

    pred = pd.read_csv(a.predictions)[["image_id", "probability"]]
    rc = reader_counts(Path(a.labels))
    df = pred.merge(rc, on="image_id", validate="one_to_one")
    df["clean_neg"] = df.nofind_readers == df.readers
    print(f"{len(df):,} scored images; readers per image: {df.readers.value_counts().to_dict()}")

    arms = {"proposed_ce": df}
    if a.resnet50:
        r50 = pd.read_csv(a.resnet50)[["image_id", "probability"]].rename(columns={"probability": "p_r50"})
        arms["baseline_resnet50"] = df.merge(r50, on="image_id", validate="one_to_one")

    print("\n--- 1. AUC by positive-label strictness (negatives = every reader said No finding) ---")
    rows = []
    for name, d in arms.items():
        col = "probability" if name == "proposed_ce" else "p_r50"
        for label, k in (("any reader (>=1)", 1), ("majority (>=2)", 2), ("unanimous (3)", 3)):
            pos = d[d.opacity_readers >= k]
            neg = d[d.clean_neg]
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            s = np.r_[pos[col].values, neg[col].values]
            lo, hi = bootstrap_auc(y, s)
            rows.append({"arm": name, "positive_definition": label, "n_pos": len(pos),
                         "n_neg": len(neg), "auc": round(roc_auc(y, s), 4),
                         "ci_lower": round(lo, 4), "ci_upper": round(hi, 4)})
    t1 = pd.DataFrame(rows)
    print(t1.to_string(index=False))

    print("\n--- 2. Ensemble AUC stratified by reader agreement on the positives ---")
    d = arms["proposed_ce"]
    neg = d[d.clean_neg]
    rows = []
    for k in (1, 2, 3):
        pos = d[d.opacity_readers == k]
        if len(pos) < 20:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos.probability.values, neg.probability.values]
        rows.append({"opacity_readers": k, "n_pos": len(pos),
                     "mean_score_pos": round(pos.probability.mean(), 4),
                     "auc_vs_clean_neg": round(roc_auc(y, s), 4)})
    t2 = pd.DataFrame(rows)
    print(t2.to_string(index=False))
    print(f"    mean score on clean negatives: {neg.probability.mean():.4f}")

    out = Path(a.out_dir) if a.out_dir else Path(a.predictions).parent
    out.mkdir(parents=True, exist_ok=True)
    t1.to_csv(out / "label_sensitivity_by_strictness.csv", index=False)
    t2.to_csv(out / "label_sensitivity_by_agreement.csv", index=False)
    print(f"\nwritten to {out}")
    print("\nIf AUC goes up as the label gets stricter, part of the drop on VinDr comes from "
          "reader disagreement rather than the model.")


if __name__ == "__main__":
    main()
