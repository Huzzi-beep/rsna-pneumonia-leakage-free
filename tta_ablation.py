from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kermany_matched_evaluation as K
import leverF_matched_evaluation as LF


def effect(y, s_tta, s_plain, thr):
    a, b = K.metrics(y, s_tta, thr), K.metrics(y, s_plain, thr)
    return {"threshold": thr, "n": int(len(y)),
            "f1_with_tta": round(100 * a["f1"], 4), "f1_without_tta": round(100 * b["f1"], 4),
            "delta_f1_pp": round(100 * (a["f1"] - b["f1"]), 4),
            "auc_with_tta": round(a["roc_auc"], 6), "auc_without_tta": round(b["roc_auc"], 6),
            "delta_auc": round(a["roc_auc"] - b["roc_auc"], 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverf-root", required=True)
    ap.add_argument("--arm", default="proposed_ce")
    ap.add_argument("--out-dir", default="tta_output")
    ap.add_argument("--threshold-min", type=float, default=0.05)
    a = ap.parse_args()
    K.THRESHOLD_GRID = np.round(np.arange(a.threshold_min, 0.8001, 0.01), 4)
    root, out = Path(a.leverf_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ens = pd.read_csv(root / f"arm_summary__{a.arm}" / "untouched_test_predictions_ensemble.csv")
    ens = ens.sort_values("image_id").reset_index(drop=True)
    y = ens.label.values
    locked = float(ens.locked_threshold.iloc[0])

    rows = [{"partition": "full untouched test split (reported in Section 4.5 / Table 13)",
             **effect(y, ens.ensemble_probability.values, ens.ensemble_probability_no_tta.values, locked)}]

    arm = LF.build_leverf_arm(root, a.arm)
    cal_ids = K.matched_partition_ids(arm["calibration"])
    test_ids = set(K.matched_partition_ids(arm["test"]))
    c = K.subset(arm["calibration"], cal_ids)
    thr = K.fit_threshold_argmax(c.label.values, c.score.values)
    m = ens[ens.image_id.isin(test_ids)]
    rows.append({"partition": "prevalence-matched test partition (for completeness)",
                 **effect(m.label.values, m.ensemble_probability.values,
                          m.ensemble_probability_no_tta.values, thr)})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"TTA ablation - arm {a.arm}\n")
    print(df.to_string(index=False))
    r = rows[0]
    print(f"\nReported: dF1 = {r['delta_f1_pp']:+.2f} pp, dAUC = {r['delta_auc']:+.4f} "
          f"(full untouched test split, n = {r['n']:,}, threshold {r['threshold']:.2f})")
    df.to_csv(out / "tta_ablation.csv", index=False)
    (out / "tta_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
