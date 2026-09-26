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

EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fuse(rule: str, mat: np.ndarray, cal_mat: np.ndarray, w: np.ndarray, best: int) -> np.ndarray:
    if rule == "AUC-weighted mean":
        return mat @ w
    if rule == "uniform mean":
        return mat.mean(axis=1)
    if rule == "rank average":
        cols = []
        for j in range(mat.shape[1]):
            ref = np.sort(cal_mat[:, j])
            cols.append(np.searchsorted(ref, mat[:, j], side="right") / len(ref))
        return np.column_stack(cols).mean(axis=1)
    if rule == "geometric mean":
        return np.exp(np.log(np.clip(mat, EPS, 1)).mean(axis=1))
    if rule == "logit mean":
        return 1 / (1 + np.exp(-logit(mat).mean(axis=1)))
    if rule == "best single member":
        return mat[:, best]
    raise ValueError(rule)


RULES = ["AUC-weighted mean", "uniform mean", "rank average",
         "geometric mean", "logit mean", "best single member"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverf-root", required=True)
    ap.add_argument("--arm", default="proposed_ce")
    ap.add_argument("--out-dir", default="fusion_output")
    ap.add_argument("--threshold-min", type=float, default=0.05)
    a = ap.parse_args()
    K.THRESHOLD_GRID = np.round(np.arange(a.threshold_min, 0.8001, 0.01), 4)
    root, out = Path(a.leverf_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(f"Fusion rules - arm {a.arm}")
    print("=" * 92)

    cal_w, test_w, names = LF.load_leverf_members(root, a.arm)
    rule_used = LF.verify_ensemble_rule(root, a.arm, test_w, names, cal_w)
    aucs = np.array([K.roc_auc(cal_w.label.values, cal_w[n].values) for n in names])
    w = aucs / aucs.sum()
    best = int(np.argmax(aucs))

    ens = pd.read_csv(root / f"arm_summary__{a.arm}" / "untouched_test_predictions_ensemble.csv")
    ens = ens.sort_values("image_id").reset_index(drop=True)
    max_diff = float(np.abs(test_w[names].values @ w - ens.ensemble_probability.values).max())
    assert max_diff < 1e-9, f"AUC-weighted scores do not reproduce the pipeline ({max_diff:.2e})"

    print(f"\nmembers: {len(names)}   pipeline rule: {rule_used}   "
          f"reproduces pipeline ensemble to {max_diff:.1e}")
    print(f"{'member':<30}{'cal AUC':>10}{'weight':>10}{'dev from 1/N':>15}")
    for n, au, wi in zip(names, aucs, w):
        print(f"{n:<30}{au:>10.4f}{wi:>10.5f}{wi - 1 / len(names):>+15.6f}")
    max_dev = float(np.max(np.abs(w - 1 / len(names))))
    print(f"max |weight - 1/N| = {max_dev:.6f}   best single member (cal AUC): {names[best]}")

    cal_frame = cal_w[["image_id", "label"]].copy(); cal_frame["score"] = 0.0
    test_frame = test_w[["image_id", "label"]].copy(); test_frame["score"] = 0.0
    cal_ids = K.matched_partition_ids(cal_frame)
    test_ids = K.matched_partition_ids(test_frame)
    cm = cal_w[cal_w.image_id.isin(set(cal_ids))].reset_index(drop=True)
    tm = test_w[test_w.image_id.isin(set(test_ids))].reset_index(drop=True)
    y_c, y_t = cm.label.values, tm.label.values
    full_cal_mat = cal_w[names].values
    print(f"\nmatched calibration n={len(cm):,}   matched test n={len(tm):,}")

    rows = []
    for rule in RULES:
        s_c = fuse(rule, cm[names].values, full_cal_mat, w, best)
        s_t = fuse(rule, tm[names].values, full_cal_mat, w, best)
        thr = K.fit_threshold_argmax(y_c, s_c)
        mc, mt = K.metrics(y_c, s_c, thr), K.metrics(y_t, s_t, thr)
        rows.append({"rule": rule, "cal_auc": round(mc["roc_auc"], 4), "threshold": thr,
                     "test_acc": round(100 * mt["accuracy"], 2), "test_prec": round(100 * mt["precision"], 2),
                     "test_rec": round(100 * mt["recall"], 2), "test_spec": round(100 * mt["specificity"], 2),
                     "test_f1": round(100 * mt["f1"], 2), "test_auc": round(mt["roc_auc"], 4),
                     "_cal_auc_exact": mc["roc_auc"], "_test_auc_exact": mt["roc_auc"],
                     "_test_f1_exact": mt["f1"]})
    df = pd.DataFrame(rows)
    df["cal_rank"] = df._cal_auc_exact.rank(ascending=False, method="min").astype(int)
    df = df.sort_values(["cal_rank", "rule"]).reset_index(drop=True)
    winner = df.iloc[0].rule

    print("\n--- ranking on the matched calibration partition ---")
    print(df[["cal_rank", "rule", "cal_auc", "threshold"]].to_string(index=False))
    print(f"\n  calibration winner: {winner}")
    print("\n--- matched test partition ---")
    print(df[["rule", "threshold", "test_acc", "test_prec", "test_rec", "test_spec",
              "test_f1", "test_auc"]].to_string(index=False))

    aw = df.set_index("rule").loc["AUC-weighted mean"]
    un = df.set_index("rule").loc["uniform mean"]
    d_f1 = 100 * (aw._test_f1_exact - un._test_f1_exact)
    d_auc = aw._test_auc_exact - un._test_auc_exact
    print(f"\n--- AUC-weighted minus uniform ---")
    print(f"  max |weight - 1/N| {max_dev:.6f}   dF1 {d_f1:+.4f} pp   dAUC {d_auc:+.2e}")

    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(
        out / "fusion_rules_matched.csv", index=False)
    pd.DataFrame({"member": names, "calibration_auc": aucs, "weight": w,
                  "deviation_from_uniform": w - 1 / len(names)}).to_csv(
        out / "fusion_member_weights.csv", index=False)
    (out / "fusion_summary.json").write_text(json.dumps({
        "arm": a.arm, "members": len(names),
        "pipeline_rule_verified": rule_used, "reproduces_pipeline_ensemble_to": max_diff,
        "max_weight_deviation_from_uniform": max_dev,
        "calibration_winner": winner,
        "auc_weighted_minus_uniform": {"delta_f1_pp": d_f1, "delta_auc": d_auc},
        "threshold_grid": [float(K.THRESHOLD_GRID[0]), float(K.THRESHOLD_GRID[-1]), 0.01],
    }, indent=2))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
