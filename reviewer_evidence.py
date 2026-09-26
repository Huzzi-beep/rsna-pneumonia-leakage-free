from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kermany_matched_evaluation as K
import leverF_matched_evaluation as F

AMBIG = "No Lung Opacity / Not Normal"


def load_manifest(results_root: Path) -> pd.DataFrame:
    run = next(results_root.glob("improved__arm-proposed__class-merged_binary__stage1-kermany*tseed42"))
    return pd.read_csv(run / "rsna_split_manifest.csv")[["image_id", "class_name"]]


def hard_class_breakdown(arms: dict, scored: dict, thr: dict, manifest: pd.DataFrame,
                         test_ids: list) -> pd.DataFrame:
    cls = manifest.set_index("image_id").loc[test_ids, "class_name"].values
    rows = []
    for label in arms:
        y, s = scored[label]
        pred = (s >= thr[label]).astype(int)
        r = {"model": label}
        for name, mask in (("Normal", cls == "Normal"), ("Ambiguous", cls == AMBIG)):
            n = int(mask.sum())
            fp = int((pred[mask] == 1).sum())
            r[f"{name}_n"] = n
            r[f"{name}_FP"] = fp
            r[f"{name}_FPR_pct"] = round(100 * fp / n, 2)
            sub = mask | (y == 1)
            r[f"{name}_AUC"] = round(K.roc_auc(y[sub], s[sub]), 4)
        r["share_of_FPs_from_ambiguous_pct"] = round(
            100 * r["Ambiguous_FP"] / max(r["Ambiguous_FP"] + r["Normal_FP"], 1), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def fixed_points(scored: dict) -> pd.DataFrame:
    rows = []
    for label, (y, s) in scored.items():
        order = np.argsort(-s, kind="mergesort")
        ys = y[order]
        tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
        P, N = (y == 1).sum(), (y == 0).sum()
        sens = tp / P; spec = 1 - fp / N
        r = {"model": label}
        for target in (0.90, 0.95):
            ok = spec >= target
            r[f"sens_at_spec{int(target*100)}"] = round(100 * (sens[ok].max() if ok.any() else 0), 2)
            ok = sens >= target
            r[f"spec_at_sens{int(target*100)}"] = round(100 * (spec[ok].max() if ok.any() else 0), 2)
        rows.append(r)
    return pd.DataFrame(rows)


def selective_prediction(scored: dict, thr: dict,
                         coverages=(1.0, 0.9, 0.8, 0.7, 0.6)) -> pd.DataFrame:
    rows = []
    for label, (y, s) in scored.items():
        conf = np.abs(s - thr[label])
        order = np.argsort(-conf)
        for cov in coverages:
            keep = order[: int(round(cov * len(y)))]
            m = K.metrics(y[keep], s[keep], thr[label])
            rows.append({"model": label, "coverage_pct": int(cov * 100),
                         "n_retained": len(keep), "n_deferred": len(y) - len(keep),
                         "accuracy": round(100 * m["accuracy"], 2),
                         "precision": round(100 * m["precision"], 2),
                         "recall": round(100 * m["recall"], 2),
                         "f1": round(100 * m["f1"], 2),
                         "auc": round(m["roc_auc"], 4)})
    return pd.DataFrame(rows)


EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fit_platt(y_cal: np.ndarray, s_cal: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    z = _logit(s_cal).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_cal)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    assert a > 0, "Platt slope must be positive for a monotone map"
    return a, b


def apply_platt(s: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _logit(s) + b)


def platt_threshold(thr_raw: float, a: float, b: float) -> float:
    return float(apply_platt(np.array([thr_raw]), a, b)[0])


def calibration(scored: dict, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, bins_out = [], []
    edges = np.linspace(0, 1, n_bins + 1)
    for label, (y, s) in scored.items():
        idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                continue
            conf, acc, n = s[m].mean(), y[m].mean(), int(m.sum())
            ece += n / len(y) * abs(acc - conf)
            bins_out.append({"model": label, "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                             "n": n, "mean_score": round(conf, 3),
                             "observed_rate": round(acc, 3), "gap": round(acc - conf, 3)})
        summary.append({"model": label, "ECE": round(ece, 4),
                        "Brier": round(float(np.mean((s - y) ** 2)), 4)})
    return pd.DataFrame(summary), pd.DataFrame(bins_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverf-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", default="reviewer_evidence_output")
    ap.add_argument("--threshold-min", type=float, default=0.05)
    a = ap.parse_args()
    lf, rr, out = Path(a.leverf_root), Path(a.results_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    K.THRESHOLD_GRID = np.round(np.arange(a.threshold_min, 0.8001, 0.01), 4)
    arms = {}
    for arm in F.LEVERF_ARMS:
        if (lf / f"arm_summary__{arm}").exists():
            arms[f"leverF {arm}"] = F.build_leverf_arm(lf, arm)
    arms["ResNet50 baseline (refit)"] = K.build_arm(rr, K.ARMS["ResNet50 baseline"])
    arms["CNN baseline"] = K.build_arm(rr, K.ARMS["CNN baseline"])
    cal_ids = K.matched_partition_ids(arms["leverF proposed_km"]["calibration"])
    test_ids = K.matched_partition_ids(arms["leverF proposed_km"]["test"])
    thr, scored = {}, {}
    for label, arm in arms.items():
        c = K.subset(arm["calibration"], cal_ids)
        thr[label] = K.fit_threshold_argmax(c.label.values, c.score.values)
        t = K.subset(arm["test"], test_ids)
        scored[label] = (t.label.values, t.score.values)

    manifest = load_manifest(rr)

    print("=" * 84)
    print("Additional analyses - prevalence-matched partition, n=1,804")
    print("=" * 84)

    hc = hard_class_breakdown(arms, scored, thr, manifest, test_ids)
    hc.to_csv(out / "01_hard_class_breakdown.csv", index=False)
    print("\n--- 1. False positives by negative subgroup ---")
    print("    The ambiguous class is 497 of the 902 negatives (55.1%).")
    print(hc.to_string(index=False))

    fp = fixed_points(scored)
    fp.to_csv(out / "02_fixed_operating_points.csv", index=False)
    print("\n--- 2. Operating characteristics at fixed points (%) ---")
    print(fp.to_string(index=False))

    sp = selective_prediction(scored, thr)
    sp.to_csv(out / "03_selective_prediction.csv", index=False)
    print("\n--- 3. Selective prediction: defer the least confident cases ---")
    print(sp[sp.model == "leverF proposed_ce"].drop(columns="model").to_string(index=False))

    scaled, invariance = {}, []
    for label, arm in arms.items():
        c = K.subset(arm["calibration"], cal_ids)
        a_, b_ = fit_platt(c.label.values, c.score.values)
        y, s = scored[label]
        s_ps = apply_platt(s, a_, b_)
        thr_ps = platt_threshold(thr[label], a_, b_)
        scaled[f"{label} + Platt"] = (y, s_ps)
        m0, m1 = K.metrics(y, s, thr[label]), K.metrics(y, s_ps, thr_ps)
        invariance.append({
            "model": label, "a": round(a_, 3), "b": round(b_, 3),
            "thr_raw": thr[label], "thr_mapped": round(thr_ps, 3),
            "auc_raw": round(m0["roc_auc"], 4), "auc_scaled": round(m1["roc_auc"], 4),
            "confusion_raw": f"{m0['tn']}/{m0['fp']}/{m0['fn']}/{m0['tp']}",
            "confusion_scaled": f"{m1['tn']}/{m1['fp']}/{m1['fn']}/{m1['tp']}",
            "identical": (m0["tn"], m0["fp"], m0["fn"], m0["tp"]) ==
                         (m1["tn"], m1["fp"], m1["fn"], m1["tp"]),
        })

    cs, cb = calibration({**scored, **scaled})
    cs.to_csv(out / "04_calibration_summary.csv", index=False)
    cb.to_csv(out / "04_calibration_bins.csv", index=False)
    inv = pd.DataFrame(invariance)
    inv.to_csv(out / "04_platt_invariance.csv", index=False)

    print("\n--- 4. Calibration, before and after Platt scaling ---")
    print(cs.to_string(index=False))
    print("\n    fitted on calibration; the map is monotone, so AUC and the confusion")
    print("    matrix (TN/FP/FN/TP) at the mapped threshold should not change:")
    print(inv.to_string(index=False))
    assert inv.identical.all(), "Platt scaling changed a decision - check the fit"
    print("\n    reliability, leverF proposed_ce before / after:")
    for lab in ("leverF proposed_ce", "leverF proposed_ce + Platt"):
        print(f"    [{lab}]")
        print(cb[cb.model == lab].drop(columns="model").to_string(index=False))

    print(f"\nAll tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
