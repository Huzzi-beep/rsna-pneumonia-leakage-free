from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="leverF_evaluation_output")
    ap.add_argument("--out-dir", default="in_text_numbers")
    a = ap.parse_args()
    ev, out = Path(a.eval_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summ = pd.read_csv(ev / "01_summary_matched.csv").set_index("model")
    rows = []

    def add(where, what, value, fmt):
        rows.append({"manuscript_location": where, "quantity": what, "value": value})
        print(f"{where:<34}{what:<78}{fmt.format(value)}")

    seeds = pd.read_csv(ev / "07_per_seed_final_vs_r50.csv")
    for tag, arm, ci_file in (("a", "ensemble (proposed_ce)", "05_ci_proposed_ce.csv"),
                              ("b", "ResNet50", "05_ci_baseline_resnet50.csv")):
        ci = pd.read_csv(ev / ci_file).set_index("metric")
        add("Section 4.1", f"{arm}: SD of F1 across the three seeds (pp)",
            100 * seeds[f"{tag}_f1"].std(ddof=1), "{:.2f}")
        add("Section 4.1", f"{arm}: SD of AUC across the three seeds",
            seeds[f"{tag}_auc"].std(ddof=1), "{:.4f}")
        add("Section 4.1", f"{arm}: half-width of the 95% bootstrap CI of F1 (pp)",
            100 * (ci.loc["f1", "ci_95_upper"] - ci.loc["f1", "ci_95_lower"]) / 2, "{:.2f}")
        add("Section 4.1", f"{arm}: half-width of the 95% bootstrap CI of AUC",
            (ci.loc["roc_auc", "ci_95_upper"] - ci.loc["roc_auc", "ci_95_lower"]) / 2, "{:.4f}")

    lf = summ[summ.index.str.startswith("leverF")]
    add("Section 4.1", "range of ROC AUC across the five corrected arms", lf.auc.max() - lf.auc.min(), "{:.4f}")

    add("Section 4.2", "ResNet50 AUC: refit on train+calibration (original) minus no refit",
        summ.loc["ResNet50 baseline", "auc"] - summ.loc["leverF baseline_resnet50", "auc"], "{:.4f}")

    nih, ker = summ.loc["Proposed (NIH stage-1)"], summ.loc["Proposed (Kermany stage-1)"]
    add("Section 3.2.4", "AUC lost by removing NIH stage-1 (NIH arm minus Kermany arm)", nih.auc - ker.auc, "{:.4f}")
    add("Section 3.2.4", "precision lost by removing NIH stage-1 (pp)", nih.prec - ker.prec, "{:.2f}")

    pd.DataFrame(rows).to_csv(out / "in_text_numbers.csv", index=False)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
