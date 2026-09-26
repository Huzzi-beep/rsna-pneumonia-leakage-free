from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kermany_matched_evaluation as K

BLUE, ORANGE, GREY = "#1f4e79", "#c55a11", "#7f7f7f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tpr = np.r_[0, np.cumsum(y == 1) / max((y == 1).sum(), 1)]
    fpr = np.r_[0, np.cumsum(y == 0) / max((y == 0).sum(), 1)]
    return fpr, tpr


def matched(df):
    ids = set(K.matched_partition_ids(df))
    return df[df.image_id.isin(ids)]


def fig4(data: Path, vindr_labels: str, out: Path):
    lf = data / "leverF_predictions"
    r50 = matched(pd.read_csv(lf / "arm_summary__baseline_resnet50" / "untouched_test_predictions_ensemble.csv"))
    ens = matched(pd.read_csv(lf / "arm_summary__proposed_ce" / "untouched_test_predictions_ensemble.csv"))
    panels = [("(a) RSNA, prevalence-matched test partition (n = 1,804)",
               [(r50.label.values, r50.ensemble_probability.values, "ResNet50"),
                (ens.label.values, ens.ensemble_probability.values, "Ensemble (CE)")])]
    ext = data / "external"
    if vindr_labels:
        lab = pd.read_csv(vindr_labels)
        clean = lab.groupby("image_id").class_name.apply(lambda s: (s == "No finding").all())
        clean_ids = set(clean[clean].index)
        cur = []
        for sub, name in (("external_validation_vindr_resnet50", "ResNet50"),
                          ("external_validation_vindr_output", "Ensemble (CE)")):
            v = pd.read_csv(ext / "vindr" / sub / "predictions_all_scored.csv")
            v = v[(v.label == 1) | v.image_id.isin(clean_ids)]
            cur.append((v.label.values, v.probability.values, name))
        panels.append((f"(b) VinDr-CXR, any reader vs. no finding (n = {len(v):,})", cur))
    cur = []
    for sub, name in (("chexpert_resnet50", "ResNet50"), ("chexpert_proposed_ce", "Ensemble (CE)")):
        c = pd.read_csv(ext / "chexpert" / sub / "predictions_all_scored.csv")
        c = c[c.label.isin([0, 1])]
        cur.append((c.label.values, c.probability.values, name))
    panels.append((f"(c) CheXpert, opacity vs. all negatives (n = {len(c):,})", cur))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.7))
    axes = np.atleast_1d(axes)
    for ax, (title, curves) in zip(axes, panels):
        for (y, s, name), col in zip(curves, (BLUE, ORANGE)):
            fpr, tpr = roc(y, s)
            ax.plot(fpr, tpr, color=col, lw=2.0, label=f"{name}  AUC {K.roc_auc(y, s):.3f}")
        ax.plot([0, 1], [0, 1], ":", color=GREY, lw=0.9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(title, fontsize=8.6, loc="left", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, frameon=False); ax.grid(alpha=.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig4_roc.png", dpi=600); fig.savefig(out / "fig4_roc.pdf"); plt.close(fig)


def fig5(eval_dir: Path, out: Path):
    cm = pd.read_csv(eval_dir / "02_confusion_matched.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (model, title) in zip(axes, (("leverF baseline_resnet50", "(a) ResNet50 (recommended)"),
                                         ("leverF proposed_ce", "(b) Ensemble (CE)"))):
        r = cm.loc[model]
        m = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1000)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]}\n({100 * m[i, j] / m[i].sum():.1f}%)", ha="center", va="center",
                        fontsize=10, color="white" if m[i, j] > 500 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Lung Opacity"])
        ax.set_yticklabels(["Negative", "Lung Opacity"], rotation=90, va="center")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f"Predicted class (threshold {r.thr:.2f})"); ax.set_ylabel("True class")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "fig5_confusion.png", dpi=600); fig.savefig(out / "fig5_confusion.pdf"); plt.close(fig)


def fig6(data: Path, out: Path):
    from PIL import Image
    files = sorted((data / "external" / "gradcam" / "examples").glob("*.png"))[:8]
    if not files:
        print("  Fig. 6 skipped: no Grad-CAM example images found")
        return
    left, right = (15, 81, 515, 580), (604, 81, 1104, 580)
    p, gap = 500, 28
    sheet = Image.new("RGB", (4 * p + 3 * gap, 4 * p + 3 * gap), "white")
    for k, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        grp, col = divmod(k, 4)
        x, y = col * (p + gap), grp * 2 * (p + gap)
        sheet.paste(im.crop(left), (x, y)); sheet.paste(im.crop(right), (x, y + p + gap))
    sheet.save(out / "fig6_gradcam.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--eval-dir", default="leverF_evaluation_output")
    ap.add_argument("--vindr-labels", default="")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig4(Path(a.data), a.vindr_labels, out); print("  Fig. 4 written")
    fig5(Path(a.eval_dir), out); print("  Fig. 5 written")
    fig6(Path(a.data), out); print("  Fig. 6 written")
    print(f"figures in {out.resolve()}")


if __name__ == "__main__":
    main()
