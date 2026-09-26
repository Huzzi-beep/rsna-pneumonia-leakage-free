from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kermany_matched_evaluation as K

SEEDS = ("42", "123", "2026")
BACKBONES = ("efficientnet_b4", "densenet121")

LEVERF_ARMS = {
    "proposed_km": BACKBONES,
    "stage1none": BACKBONES,
    "loss_ce": BACKBONES,
    "baseline_resnet50": ("resnet50",),
    "proposed_ce": BACKBONES,
}


def load_leverf_members(root: Path, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cal_w = test_w = None
    names = []
    for bb in LEVERF_ARMS[arm]:
        for seed in SEEDS:
            d = root / f"leverF__{arm}__{bb}__seed{seed}"
            if not d.exists():
                raise FileNotFoundError(d)
            name = f"{bb}__seed{seed}"
            names.append(name)
            c = pd.read_csv(d / "calibration_predictions.csv")[["image_id", "label", "probability"]]
            t = pd.read_csv(d / "untouched_test_predictions.csv")[["image_id", "label", "probability"]]
            c = c.rename(columns={"probability": name})
            t = t.rename(columns={"probability": name})
            cal_w = c if cal_w is None else cal_w.merge(c, on=["image_id", "label"], validate="one_to_one")
            test_w = t if test_w is None else test_w.merge(t, on=["image_id", "label"], validate="one_to_one")
    return (cal_w.sort_values("image_id").reset_index(drop=True),
            test_w.sort_values("image_id").reset_index(drop=True), names)


def verify_ensemble_rule(root: Path, arm: str, test_w: pd.DataFrame, names: list[str],
                         cal_w: pd.DataFrame) -> str:
    ens = pd.read_csv(root / f"arm_summary__{arm}" / "untouched_test_predictions_ensemble.csv")
    ens = ens.sort_values("image_id").reset_index(drop=True)
    assert np.array_equal(ens.image_id.values, test_w.image_id.values)
    ref = ens.ensemble_probability.values

    uniform = test_w[names].values.mean(axis=1)
    aucs = np.array([K.roc_auc(cal_w.label.values, cal_w[n].values) for n in names])
    weighted = test_w[names].values @ (aucs / aucs.sum())

    du, dw = np.abs(uniform - ref).max(), np.abs(weighted - ref).max()
    rule = "uniform" if du < dw else "auc-weighted"
    print(f"  {arm:<12} ensemble rule = {rule:<12} "
          f"(max |diff| uniform {du:.2e}, auc-weighted {dw:.2e})")
    return rule


def build_leverf_arm(root: Path, arm: str) -> dict:
    cal_w, test_w, names = load_leverf_members(root, arm)
    rule = verify_ensemble_rule(root, arm, test_w, names, cal_w)
    aucs = np.array([K.roc_auc(cal_w.label.values, cal_w[n].values) for n in names])
    w = (np.full(len(names), 1 / len(names)) if rule == "uniform" else aucs / aucs.sum())
    cal = cal_w[["image_id", "label"]].copy();   cal["score"] = cal_w[names].values @ w
    test = test_w[["image_id", "label"]].copy(); test["score"] = test_w[names].values @ w
    return {"calibration": cal, "test": test, "members": names,
            "cal_w": cal_w, "test_w": test_w, "member_auc": dict(zip(names, aucs))}


def pct(x): return round(100 * x, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverf-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", default="leverF_evaluation_output")
    ap.add_argument("--threshold-min", type=float, default=0.05,
                    help="lowest threshold searched, same for all arms. The first version used 0.20, "
                         "but the CE arms peak below that, so the paper uses 0.05.")
    a = ap.parse_args()
    K.THRESHOLD_GRID = np.round(np.arange(a.threshold_min, 0.8001, 0.01), 4)
    print(f"threshold grid: {K.THRESHOLD_GRID[0]:.2f}-{K.THRESHOLD_GRID[-1]:.2f} step 0.01")
    lf, rr, out = Path(a.leverf_root), Path(a.results_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("leverF arms on the prevalence-matched partition")
    print("=" * 84)

    print("\n--- 1. Ensemble construction ---")
    arms = {}
    for arm in LEVERF_ARMS:
        d = lf / f"arm_summary__{arm}"
        if d.exists():
            arms[f"leverF {arm}"] = build_leverf_arm(lf, arm)
    for label, spec in K.ARMS.items():
        arms[label] = K.build_arm(rr, spec)
        print(f"  {label:<28} {len(arms[label]['members'])} members (original run family)")

    ref = arms["leverF proposed_km"]
    cal_ids = K.matched_partition_ids(ref["calibration"])
    test_ids = K.matched_partition_ids(ref["test"])
    for label, arm in arms.items():
        assert K.matched_partition_ids(arm["test"]) == test_ids, f"{label}: partition mismatch"
    print(f"\n  matched calibration {len(cal_ids):,} | matched test {len(test_ids):,} | "
          f"identical across all {len(arms)} arms")

    print("\n--- 2. Thresholds (argmax F1, matched calibration only) ---")
    thr, scored = {}, {}
    for label, arm in arms.items():
        c = K.subset(arm["calibration"], cal_ids)
        thr[label] = K.fit_threshold_argmax(c.label.values, c.score.values)
        t = K.subset(arm["test"], test_ids)
        scored[label] = (t.label.values, t.score.values)
        print(f"  {label:<28} {thr[label]:.2f}")

    rows, cms = [], []
    for label in arms:
        y, s = scored[label]
        m = K.metrics(y, s, thr[label])
        rows.append({"model": label, "acc": pct(m["accuracy"]), "prec": pct(m["precision"]),
                     "rec": pct(m["recall"]), "spec": pct(m["specificity"]), "f1": pct(m["f1"]),
                     "auc": round(m["roc_auc"], 4), "auprc": round(m["auprc"], 4),
                     "mcc": round(m["mcc"], 4)})
        cms.append({"model": label, "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"],
                    "thr": thr[label]})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "01_summary_matched.csv", index=False)
    pd.DataFrame(cms).to_csv(out / "02_confusion_matched.csv", index=False)
    print("\n--- 3. Matched-partition summary ---")
    print(summary.to_string(index=False))
    print("\n  Usman et al. (2025), published: 79.00  76.00  73.00   --    74.00  0.8500")
    print("\n--- 4. Confusion matrices ---")
    print(pd.DataFrame(cms).to_string(index=False))

    for label in [l for l in arms if l.startswith("leverF")]:
        y, s = scored[label]
        ci = K.bootstrap_ci(y, s, thr[label])
        ci.to_csv(out / f"05_ci_{label.split()[-1]}.csv", index=False)
        print(f"\n--- 5. Bootstrap 95% CI: {label} ---")
        print(ci.round(4).to_string(index=False))

    def paired(a_label: str, b_label: str, tag: str):
        y_a, s_a = scored[a_label]; y_b, s_b = scored[b_label]
        assert np.array_equal(y_a, y_b)
        pb = K.paired_bootstrap(y_a, s_a, thr[a_label], s_b, thr[b_label])
        mc = K.mcnemar(y_a, s_a, thr[a_label], s_b, thr[b_label])
        dl = K.delong(y_a, s_a, s_b)
        key = f"{a_label.split()[-1]}_vs_{b_label.replace(' ', '_').replace('(', '').replace(')', '')}"
        pb.to_csv(out / f"06_paired_{key}.csv", index=False)
        (out / f"06_tests_{key}.json").write_text(json.dumps({"mcnemar": mc, "delong": dl}, indent=2))
        print(f"\n--- 6. {tag}: {a_label} vs {b_label} ---")
        print(pb.round(4).to_string(index=False))
        print(f"  McNemar b={mc['b']} c={mc['c']} chi2={mc['chi2']:.3f} p={mc['p_value']:.4f}")
        print(f"  DeLong  z={dl['z']:.3f} p={dl['p_value']:.4f}")

    km, ce, r50 = "leverF proposed_km", "leverF loss_ce", "leverF baseline_resnet50"
    fin = "leverF proposed_ce"
    if fin in arms:
        if r50 in arms:
            paired(fin, r50, "final ensemble vs ResNet50, same protocol")
        if ce in arms:
            paired(fin, ce, "stage-1 with CE loss: none vs Kermany")
        paired(fin, km, "final ensemble vs focal + Kermany arm")
    paired(km, "leverF stage1none", "stage-1 Kermany vs none")
    if ce in arms:
        paired(ce, km, "CE + LS vs focal + LS")
    if r50 in arms:
        paired(km, r50, "focal arm vs ResNet50, same protocol")
        if ce in arms:
            paired(ce, r50, "CE arm vs ResNet50, same protocol")
    paired(km, "ResNet50 baseline", "vs first-round ResNet50 (refit, not like for like)")
    paired(km, "CNN baseline", "vs CNN")
    paired(km, "Proposed (Kermany stage-1)", "vs first-round Kermany arm (refit)")

    def per_seed_test(a_label, b_label, title, section):
        A, B = arms[a_label], arms[b_label]
        rows = []
        for seed in SEEDS:
            rec = {"seed": seed}
            for tag, arm in (("a", A), ("b", B)):
                cols = [n for n in arm["members"] if n.endswith(f"seed{seed}")]
                c = K.subset(arm["cal_w"], cal_ids); t = K.subset(arm["test_w"], test_ids)
                sc, st = c[cols].values.mean(1), t[cols].values.mean(1)
                th = K.fit_threshold_argmax(c.label.values, sc)
                m = K.metrics(t.label.values, st, th)
                rec[f"{tag}_auc"], rec[f"{tag}_f1"], rec[f"{tag}_acc"] = m["roc_auc"], m["f1"], m["accuracy"]
            rows.append(rec)
        ps = pd.DataFrame(rows)
        ps.to_csv(out / f"07_per_seed_{section}.csv", index=False)
        print(f"\n--- 7. {title} (per seed; a = {a_label}, b = {b_label}) ---")
        print(ps.round(4).to_string(index=False))
        for metric in ("auc", "f1", "acc"):
            d = ps[f"a_{metric}"].values - ps[f"b_{metric}"].values
            tt = stats.ttest_rel(ps[f"a_{metric}"], ps[f"b_{metric}"])
            try:
                wp = f"{stats.wilcoxon(d).pvalue:.3f}"
            except ValueError:
                wp = "n/a"
            print(f"  {metric:<4} mean diff {d.mean():+.4f} (sd {d.std(ddof=1):.4f})  "
                  f"paired t p={tt.pvalue:.3f}  wilcoxon p={wp}")

    per_seed_test(km, "leverF stage1none", "stage-1 vs none", "r4_2")
    if ce in arms:
        per_seed_test(ce, km, "CE vs focal", "r4_4")
    if r50 in arms and ce in arms:
        per_seed_test(ce, r50, "CE arm vs ResNet50", "ce_vs_r50")
    if fin in arms and r50 in arms:
        per_seed_test(fin, r50, "final ensemble vs ResNet50", "final_vs_r50")

    print(f"\nAll tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
