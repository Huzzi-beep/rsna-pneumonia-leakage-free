from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kermany_matched_evaluation as K

OPACITY = {"Lung Opacity", "Consolidation", "Infiltration"}
CHEX_OPACITY = ("Lung Opacity", "Consolidation", "Pneumonia")


def vindr_frames(inf: Path, labels_csv: Path | None = None):
    lab = pd.read_csv(labels_csv or (inf / "vindr_train_labels.csv"))
    lab["op"] = lab.class_name.isin(OPACITY)
    lab["nf"] = lab.class_name == "No finding"
    g = lab.groupby("image_id")
    per = pd.DataFrame({"readers": g.rad_id.nunique(),
                        "op_readers": g.apply(lambda d: d.loc[d.op, "rad_id"].nunique()),
                        "nf_readers": g.apply(lambda d: d.loc[d.nf, "rad_id"].nunique())}).reset_index()
    per["clean_neg"] = per.nf_readers == per.readers
    a = pd.read_csv(inf / "external_validation_vindr_output" / "predictions_all_scored.csv")[["image_id", "probability"]]
    b = pd.read_csv(inf / "external_validation_vindr_resnet50" / "predictions_all_scored.csv")[["image_id", "probability"]]
    df = per.merge(a, on="image_id").merge(b.rename(columns={"probability": "p_b"}), on="image_id")
    out = {}
    for name, k in (("VinDr · any reader", 1), ("VinDr · majority", 2), ("VinDr · unanimous", 3)):
        sel = df[(df.op_readers >= k) | df.clean_neg]
        y = (sel.op_readers >= k).astype(int).values
        out[name] = (y, sel.probability.values, sel.p_b.values)
    return out


def chex_frames(chex: Path):
    a = pd.read_csv(chex / "chexpert_proposed_ce" / "predictions_all_scored.csv")
    b = pd.read_csv(chex / "chexpert_resnet50" / "predictions_all_scored.csv")[["image_id", "probability"]]
    df = a.merge(b.rename(columns={"probability": "p_b"}), on="image_id", validate="one_to_one")
    y = df.label.values
    return {"CheXpert · opacity vs all negatives": (y, df.probability.values, df.p_b.values)}


def compare(name, y, sa, sb, rows):
    dl = K.delong(y, sa, sb)
    rng = np.random.default_rng(42)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(K.roc_auc(y[i], sa[i]) - K.roc_auc(y[i], sb[i]))
    d = np.asarray(d)
    rows.append({"dataset / definition": name, "n": len(y),
                 "ensemble AUC": round(dl["auc_a"], 4), "ResNet50 AUC": round(dl["auc_b"], 4),
                 "difference": round(dl["auc_a"] - dl["auc_b"], 4),
                 "95% CI low": round(np.percentile(d, 2.5), 4),
                 "95% CI high": round(np.percentile(d, 97.5), 4),
                 "DeLong p": f"{dl['p_value']:.2e}" if dl["p_value"] < 1e-3 else f"{dl['p_value']:.3f}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inf", required=True)
    ap.add_argument("--chex", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--vindr-labels", default="", help="VinDr-CXR train.csv (default: <inf>/vindr_train_labels.csv)")
    a = ap.parse_args()
    rows = []
    for name, (y, sa, sb) in vindr_frames(Path(a.inf), Path(a.vindr_labels) if a.vindr_labels else None).items():
        compare(name, y, sa, sb, rows)
    for name, (y, sa, sb) in chex_frames(Path(a.chex)).items():
        compare(name, y, sa, sb, rows)
    t = pd.DataFrame(rows)
    print("Paired AUC comparison on the same images: ensemble (proposed_ce) minus ResNet50\n")
    print(t.to_string(index=False))
    print("\nFor reference, in-distribution (RSNA matched partition): +0.0046, DeLong p = 0.099")
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
