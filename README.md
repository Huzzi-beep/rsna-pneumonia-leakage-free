# Pneumonia detection on RSNA chest radiographs under a leakage-free evaluation protocol

Code for:

> *Pneumonia Detection on RSNA Chest Radiographs under a Leakage-Free Evaluation Protocol:
> A Calibrated Single ResNet50 Generalizes Better than a Cross-Domain Ensemble*

The repository contains the training pipeline, the fixed data split, and the scripts for the
evaluation, statistical tests, external validation, Grad-CAM localisation and the NIH / RSNA
overlap audit. No images are included; the datasets are available from their providers.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the evaluation packages with the versions we used. Training and the
GPU scripts also need `torch`, `torchvision`, `timm`, `albumentations`, `opencv-python` and
`pydicom`.

## Data split

`rsna_split_manifest.csv` is the one stratified split used for every model (seed 42):
70% train, 15% calibration, 15% untouched test (MD5 `2ebe4d5a2a4846c32f009636103af679`).

## 1. Training (GPU)

Run from the repository folder:

```bash
python leverF_pipeline.py --stage cache --manifest-csv rsna_split_manifest.csv \
    --cache-dir /tmp/png_cache_f --rsna-root <RSNA competition folder>

python run_chunk_F.py --chunk S  --manifest-csv rsna_split_manifest.csv \
    --cache-dir /tmp/png_cache_f --kermany-root <Kermany chest_xray folder>
python run_chunk_F.py --chunk F4 --manifest-csv rsna_split_manifest.csv --cache-dir /tmp/png_cache_f
```

| Chunk | Arm | Settings |
|---|---|---|
| S | Kermany stage-1 weights | needed by F1 and F3 |
| F1 | `proposed_km` | ensemble, Kermany stage-1, focal loss + label smoothing |
| F2 | `stage1none` | ensemble, no stage-1, focal loss + label smoothing |
| F3 | `loss_ce` | ensemble, Kermany stage-1, cross-entropy + label smoothing |
| R | `baseline_resnet50` | ResNet50, no stage-1, cross-entropy + label smoothing (recommended model) |
| F4 | `proposed_ce` | ensemble, no stage-1, cross-entropy + label smoothing |

Every model is trained on the 70% split only, with three seeds (42, 123, 2026); the
calibration split is used for early stopping, ensemble weights and the decision threshold,
and there is no refit. After each chunk, combine the members of the arm:

```bash
python leverF_pipeline.py --stage evaluate --arm proposed_ce \
    --out-root <results folder> --cache-dir /tmp/png_cache_f --manifest-csv rsna_split_manifest.csv
```

GPU training is not bit-for-bit deterministic, so retrained models will differ slightly
from ours.

## 2. Results

| Paper | Script |
|---|---|
| Tables 8-12, per-seed tests (Sections 4.1-4.4), Fig. 5 | `leverF_matched_evaluation.py` |
| Table 13 - fusion rules | `fusion_search.py` |
| Table 13 - test-time augmentation | `tta_ablation.py` |
| Tables 17-20 | `reviewer_evidence.py` |
| Numbers in the text (Sections 3.2.4, 4.1, 4.2) | `in_text_numbers.py` |
| Table 14 - external validation (GPU) | `external_validation_vindr.py`, `external_validation_chexpert.py` |
| Table 15 - paired external comparison | `external_paired_comparison.py` |
| Table 16 - VinDr reference-standard sensitivity | `vindr_label_sensitivity.py` |
| Table 21, Fig. 6 - Grad-CAM localisation (GPU) | `gradcam_localization.py` |
| Figs. 4-6 | `make_figures.py` |
| Section 3.2.4 - NIH / RSNA overlap audit | `nih_rsna_overlap_audit.py` |

`kermany_matched_evaluation.py` holds the metric, threshold, partition and statistical-test
functions the other scripts use. Each script prints its usage with `--help`. The scripts take
as input the prediction files that the training pipeline writes for each model.

Protocol: the headline metrics use the prevalence-matched partition of the test split (all
902 positives and the first 902 negatives by image id), with the threshold at the F1 maximum
on the matched calibration partition (grid 0.05-0.80). Figures 1-3 are diagrams; Table 22
(inference time) is a hardware measurement.

The pipeline's default for `--stage1` is `none`: NIH ChestX-ray14 contains the RSNA images
(`nih_rsna_overlap_audit.py`), so NIH pretraining is not used for the reported models.

## Datasets

RSNA Pneumonia Detection Challenge, Kermany et al. (paediatric chest X-rays), NIH
ChestX-ray14, VinDr-CXR and CheXpert are available from their providers under their own
terms.

## Citation

[citation to be added on publication] - archived at Zenodo: [DOI]
