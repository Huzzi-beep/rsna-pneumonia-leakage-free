from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

THRESHOLD_GRID = np.round(np.arange(0.20, 0.8001, 0.01), 4)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
N_THRESHOLD_BOOTSTRAP = 500
SEEDS = ["42", "123", "2026"]

ARMS = {
    "CNN baseline": {
        "key": "baseline_cnn",
        "prefix": "improved__arm-baseline_cnn__class-merged_binary",
        "columns": ["probability_simple_cnn"],
        "ensemble_dir": "seedensemble__arm-baseline_cnn__class-merged_binary__split42",
    },
    "ResNet50 baseline": {
        "key": "baseline_resnet50",
        "prefix": "improved__arm-baseline_resnet50__class-merged_binary",
        "columns": ["probability_resnet50"],
        "ensemble_dir": "seedensemble__arm-baseline_resnet50__class-merged_binary__split42",
    },
    "Proposed (Kermany stage-1)": {
        "key": "proposed_kermany",
        "prefix": "improved__arm-proposed__class-merged_binary__stage1-kermany",
        "columns": ["probability_densenet121", "probability_efficientnet_b4"],
        "ensemble_dir": None,
    },
    "Proposed (NIH stage-1)": {
        "key": "proposed_nih",
        "prefix": "leverD__arm-proposed__class-merged_binary__stage1-nih",
        "columns": ["probability_densenet121", "probability_efficientnet_b4"],
        "ensemble_dir": "seedensemble__arm-proposed__class-merged_binary__split42",
    },
}

CONTAMINATED = {"proposed_nih"}


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = stats.rankdata(s)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max((y == 1).sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def confusion(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tn, fp, fn, tp


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    tn, fp, fn, tp = confusion(y, s, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "specificity": spec, "f1": f1, "mcc": float(mcc),
        "roc_auc": roc_auc(y, s), "auprc": average_precision(y, s),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": int(len(y)),
    }


def _f1_grid(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    pred = s[None, :] >= THRESHOLD_GRID[:, None]
    pos = (y == 1)[None, :]
    tp = np.sum(pred & pos, axis=1).astype(float)
    fp = np.sum(pred & ~pos, axis=1).astype(float)
    fn = np.sum(~pred & pos, axis=1).astype(float)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)


def fit_threshold_argmax(y: np.ndarray, s: np.ndarray) -> float:
    return float(THRESHOLD_GRID[int(np.argmax(_f1_grid(y, s)))])


def fit_threshold_bootstrap_median(y: np.ndarray, s: np.ndarray,
                                   n_boot: int = N_THRESHOLD_BOOTSTRAP,
                                   seed: int = BOOTSTRAP_SEED) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    picks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        picks.append(float(THRESHOLD_GRID[int(np.argmax(_f1_grid(yb, sb)))]))
    return float(np.median(picks))


def _run_dir(root: Path, prefix: str, seed: str) -> Path:
    matches = sorted(root.glob(f"{prefix}*tseed{seed}"))
    if not matches:
        raise FileNotFoundError(f"no run directory matching {prefix!r} seed {seed} under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous run directories for {prefix!r} seed {seed}: {matches}")
    return matches[0]


def build_arm(root: Path, spec: dict) -> dict:
    cal_members, test_members, names = {}, {}, []

    for seed in SEEDS:
        run = _run_dir(root, spec["prefix"], seed)
        cal = pd.read_csv(run / "calibration_predictions_all_models.csv")
        test = pd.read_csv(run / "untouched_test_predictions_all_models.csv")
        cal = cal.sort_values("image_id").reset_index(drop=True)
        test = test.sort_values("image_id").reset_index(drop=True)

        for col in spec["columns"]:
            name = f"{run.name}::{col}"
            names.append(name)
            cal_members[name] = cal[["image_id", "label", col]].rename(columns={col: name})
            test_members[name] = test[["image_id", "label", col]].rename(columns={col: name})

    def merge(members: dict) -> pd.DataFrame:
        out = members[names[0]]
        for n in names[1:]:
            out = out.merge(members[n], on=["image_id", "label"], validate="one_to_one")
        return out.sort_values("image_id").reset_index(drop=True)

    cal_df, test_df = merge(cal_members), merge(test_members)

    y_cal = cal_df.label.values
    aucs = np.array([roc_auc(y_cal, cal_df[n].values) for n in names])
    weights = aucs / aucs.sum()

    cal_mat = cal_df[names].values
    test_mat = test_df[names].values

    cal_out = cal_df[["image_id", "label"]].copy()
    test_out = test_df[["image_id", "label"]].copy()
    cal_out["score"] = cal_mat @ weights
    test_out["score"] = test_mat @ weights
    cal_out["score_uniform"] = cal_mat.mean(axis=1)
    test_out["score_uniform"] = test_mat.mean(axis=1)

    members = pd.DataFrame({"member": names, "calibration_auc": aucs, "weight": weights})
    return {"calibration": cal_out, "test": test_out, "members": members}


def verify_weights(root: Path, spec: dict, members: pd.DataFrame) -> str:
    if spec["ensemble_dir"] is None:
        return "no members.csv on disk (ensemble built by this script)"
    path = root / spec["ensemble_dir"] / "members.csv"
    if not path.exists():
        return "members.csv not found"
    ref = pd.read_csv(path).set_index("member")
    ours = members.set_index("member")
    common = ref.index.intersection(ours.index)
    if len(common) != len(ref):
        return f"member sets differ (matched {len(common)}/{len(ref)})"
    dw = float(np.max(np.abs(ref.loc[common, "weight"].values - ours.loc[common, "weight"].values)))
    da = float(np.max(np.abs(ref.loc[common, "calibration_auc"].values
                             - ours.loc[common, "calibration_auc"].values)))
    return f"verified against members.csv (max |dweight| {dw:.2e}, max |dAUC| {da:.2e})"


def matched_partition_ids(df: pd.DataFrame) -> list[str]:
    pos = sorted(df.loc[df.label == 1, "image_id"])
    neg = sorted(df.loc[df.label == 0, "image_id"])
    if len(neg) < len(pos):
        raise ValueError("fewer negatives than positives; cannot match prevalence")
    return sorted(pos + neg[:len(pos)])


def subset(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[df.image_id.isin(set(ids))].sort_values("image_id").reset_index(drop=True)


def bootstrap_ci(y: np.ndarray, s: np.ndarray, thr: float,
                 n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "mcc"]
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = metrics(yb, sb, thr)
        for k in keys:
            draws[k].append(m[k])
    point = metrics(y, s, thr)
    return pd.DataFrame([
        {"metric": k, "point_estimate": point[k],
         "ci_95_lower": float(np.percentile(v, 2.5)),
         "ci_95_upper": float(np.percentile(v, 97.5))}
        for k, v in draws.items()
    ])


def mcnemar(y: np.ndarray, sa: np.ndarray, ta: float,
            sb: np.ndarray, tb: float) -> dict:
    a_ok = (sa >= ta).astype(int) == y
    b_ok = (sb >= tb).astype(int) == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1))}


def delong(y: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> dict:
    m = int((y == 1).sum())
    n = int((y == 0).sum())

    def structural(scores):
        p, q = scores[y == 1], scores[y == 0]
        v10 = np.array([(np.sum(q < x) + 0.5 * np.sum(q == x)) / n for x in p])
        v01 = np.array([(np.sum(p > x) + 0.5 * np.sum(p == x)) / m for x in q])
        return v10, v01

    a10, a01 = structural(sa)
    b10, b01 = structural(sb)
    auc_a, auc_b = a10.mean(), b10.mean()

    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(auc_a), "auc_b": float(auc_b),
                "z": float("nan"), "p_value": float("nan")}
    z = (auc_a - auc_b) / np.sqrt(var)
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def paired_bootstrap(y: np.ndarray, sa: np.ndarray, ta: float,
                     sb: np.ndarray, tb: float,
                     n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    diffs: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ma = metrics(yb, sa[idx], ta)
        mb = metrics(yb, sb[idx], tb)
        for k in keys:
            diffs[k].append(ma[k] - mb[k])

    point_a, point_b = metrics(y, sa, ta), metrics(y, sb, tb)
    rows = []
    for k, v in diffs.items():
        v = np.asarray(v)
        tail = min((v <= 0).mean(), (v >= 0).mean())
        rows.append({
            "metric": k, "arm_a": point_a[k], "arm_b": point_b[k],
            "difference": point_a[k] - point_b[k],
            "ci_95_lower": float(np.percentile(v, 2.5)),
            "ci_95_upper": float(np.percentile(v, 97.5)),
            "bootstrap_p": float(min(1.0, 2 * tail)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="path to rsna_leakage_free_results")
    ap.add_argument("--out-dir", default="kermany_evaluation_output")
    ap.add_argument("--threshold-rule", choices=["argmax", "bootstrap-median"], default="argmax",
                    help="how to pick the threshold (same for every arm). argmax is deterministic; "
                         "bootstrap-median was the original rule and can be one grid step off. "
                         "Both are printed.")
    args = ap.parse_args()

    def fit(y, s):
        return (fit_threshold_argmax(y, s) if args.threshold_rule == "argmax"
                else fit_threshold_bootstrap_median(y, s))

    root, out = Path(args.results_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("Reference arms on the prevalence-matched partition")
    print("=" * 84)

    arms = {}
    print("\n--- 1. Ensemble construction ---")
    for label, spec in ARMS.items():
        arms[label] = build_arm(root, spec)
        note = verify_weights(root, spec, arms[label]["members"])
        flag = "  [CONTAMINATED: 62.6% of test seen in stage-1]" if spec["key"] in CONTAMINATED else ""
        print(f"  {label:<28} {len(arms[label]['members'])} members   {note}{flag}")
        arms[label]["members"].to_csv(out / f"01_members_{spec['key']}.csv", index=False)

    ref = arms["Proposed (Kermany stage-1)"]
    cal_ids = matched_partition_ids(ref["calibration"])
    test_ids = matched_partition_ids(ref["test"])
    print(f"\n  Matched calibration partition : {len(cal_ids):,} cases")
    print(f"  Matched test partition        : {len(test_ids):,} cases")

    print("\n--- 2. Thresholds (matched calibration partition) ---")
    print(f"  rule in force: {args.threshold_rule}")
    print(f"  {'arm':<28} {'in force':>10} {'boot-median':>12} {'argmax':>8}")
    thresholds = {}
    for label in ARMS:
        c = subset(arms[label]["calibration"], cal_ids)
        y_c, s_c = c.label.values, c.score.values
        boot = fit_threshold_bootstrap_median(y_c, s_c)
        amax = fit_threshold_argmax(y_c, s_c)
        thresholds[label] = fit(y_c, s_c)
        print(f"  {label:<28} {thresholds[label]:>10.2f} {boot:>12.2f} {amax:>8.2f}")

    summary, matrices, scored = [], [], {}
    for label, spec in ARMS.items():
        t = subset(arms[label]["test"], test_ids)
        y, s = t.label.values, t.score.values
        scored[label] = (y, s)
        m = metrics(y, s, thresholds[label])
        summary.append({
            "model": label, "accuracy_pct": round(100 * m["accuracy"], 2),
            "precision_pct": round(100 * m["precision"], 2),
            "recall_pct": round(100 * m["recall"], 2),
            "f1_pct": round(100 * m["f1"], 2),
            "specificity_pct": round(100 * m["specificity"], 2),
            "roc_auc": round(m["roc_auc"], 4), "auprc": round(m["auprc"], 4),
            "mcc": round(m["mcc"], 4),
        })
        matrices.append({"model": label, "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
                         "TP": m["tp"], "N": m["n"],
                         "threshold": round(thresholds[label], 2)})

    pd.DataFrame(summary).to_csv(out / "02_model_summary_matched.csv", index=False)
    pd.DataFrame(matrices).to_csv(out / "03_confusion_matrices_matched.csv", index=False)
    print("\n--- 3. Model summary, prevalence-matched test partition ---")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n--- 4. Confusion matrices ---")
    print(pd.DataFrame(matrices).to_string(index=False))

    for label in ("Proposed (Kermany stage-1)", "Proposed (NIH stage-1)"):
        y, s = scored[label]
        ci = bootstrap_ci(y, s, thresholds[label])
        ci.to_csv(out / f"05_bootstrap_ci_{ARMS[label]['key']}.csv", index=False)
        print(f"\n--- 5. Bootstrap 95% CI - {label} ---")
        print(ci.round(4).to_string(index=False))

    kerm = "Proposed (Kermany stage-1)"
    y_k, s_k = scored[kerm]
    for other in ("ResNet50 baseline", "CNN baseline", "Proposed (NIH stage-1)"):
        y_o, s_o = scored[other]
        assert np.array_equal(y_k, y_o), "arms are not aligned on the same cases"
        pb = paired_bootstrap(y_k, s_k, thresholds[kerm], s_o, thresholds[other])
        mc = mcnemar(y_k, s_k, thresholds[kerm], s_o, thresholds[other])
        dl = delong(y_k, s_k, s_o)
        tag = ARMS[other]["key"]
        pb.to_csv(out / f"06_paired_kermany_vs_{tag}.csv", index=False)
        with open(out / f"06_tests_kermany_vs_{tag}.json", "w") as fh:
            json.dump({"mcnemar": mc, "delong": dl}, fh, indent=2)
        print(f"\n--- 6. Proposed (Kermany) vs {other} ---")
        print(pb.round(4).to_string(index=False))
        print(f"  McNemar : b={mc['b']} c={mc['c']} chi2={mc['chi2']:.4f} p={mc['p_value']:.4e}")
        print(f"  DeLong  : z={dl['z']:.4f} p={dl['p_value']:.4e}")

    print("\n--- 7. AUC-weighted vs uniform (1/N) fusion ---")
    a2 = []
    for label in ("Proposed (Kermany stage-1)", "Proposed (NIH stage-1)"):
        t = subset(arms[label]["test"], test_ids)
        y = t.label.values
        w = metrics(y, t.score.values, thresholds[label])
        u = metrics(y, t.score_uniform.values, thresholds[label])
        wts = arms[label]["members"].weight.values
        a2.append({
            "arm": label, "n_members": len(wts),
            "max_weight_deviation_from_uniform": round(float(np.max(np.abs(wts - 1 / len(wts)))), 6),
            "f1_auc_weighted": round(100 * w["f1"], 4),
            "f1_uniform": round(100 * u["f1"], 4),
            "f1_difference_pp": round(100 * (w["f1"] - u["f1"]), 4),
            "auc_weighted": round(w["roc_auc"], 6),
            "auc_uniform": round(u["roc_auc"], 6),
            "auc_difference": round(w["roc_auc"] - u["roc_auc"], 6),
        })
    pd.DataFrame(a2).to_csv(out / "07_a2_weighted_vs_uniform.csv", index=False)
    print(pd.DataFrame(a2).to_string(index=False))

    print("\n--- 8. Full test split, natural prevalence ---")
    supp = []
    for label in ARMS:
        cal_full = arms[label]["calibration"]
        test_full = arms[label]["test"]
        thr = fit(cal_full.label.values, cal_full.score.values)
        m = metrics(test_full.label.values, test_full.score.values, thr)
        supp.append({
            "model": label, "threshold": round(thr, 2),
            "accuracy_pct": round(100 * m["accuracy"], 2),
            "precision_pct": round(100 * m["precision"], 2),
            "recall_pct": round(100 * m["recall"], 2),
            "f1_pct": round(100 * m["f1"], 2),
            "specificity_pct": round(100 * m["specificity"], 2),
            "roc_auc": round(m["roc_auc"], 4), "auprc": round(m["auprc"], 4),
            "mcc": round(m["mcc"], 4), "n": m["n"],
        })
    pd.DataFrame(supp).to_csv(out / "08_full_split_native_prevalence.csv", index=False)
    print(pd.DataFrame(supp).to_string(index=False))

    print(f"\nAll tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
