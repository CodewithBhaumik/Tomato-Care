# Model card — Tomato Care

Format after Mitchell et al., *Model Cards for Model Reporting* (2019). Numbers: [RESULTS.md](RESULTS.md).

## Model details

* **What**: image classifiers for single tomato leaves photographed on plain paper. Two served models (one per mode),
  both version 2; the calibrated focused variant (`calibrated_5class`) is evaluated but not served:
  * **Focused · balanced** (`balanced_5class`, default): Healthy,
    Leaf Miner, Magnesium Deficiency, Nitrogen Deficiency, Spotted Wilt Virus — the classes with enough photographs
    to be recognised reliably.
  * **Full · balanced** (`balanced_8class`): the five above plus Early Blight, Late Blight and Potassium Deficiency.
  * Both post-process the fold networks `upgrade_5class` / `upgrade_8class` with per-class calibration, and the
    balanced ones also with a class-prior correction (`scripts/31_balance_models.py`, fitted on out-of-fold data).
* **Architecture**: fold ensembles of ConvNeXt-Tiny (ImageNet-1k pretrained) fine-tuned at 256×256, one view per
  network — the focused model is the 5 cross-validation networks of the robust recipe; the full model the 5 of the
  original recipe plus the 5 of the robust recipe (members, temperatures and thresholds:
  `experiments/deployment.json`). Probabilities are temperature-scaled per member and averaged.
* **Outputs**: calibrated class probabilities; a decision — *Healthy*, *Diseased*, *Uncertain* (confidence below a
  threshold fitted on out-of-fold data) or *Unusual image* (feature-space distance to the training photos above a
  threshold fitted on out-of-fold data); Grad-CAM maps on request.
* **Version**: 2.0 (upgrade of the original student project). Code: MIT licence.

## Intended use

* Decision support and education: a second opinion on a leaf photo, explaining what the network looked at.
* Teaching material for leakage-free evaluation, calibration and selective prediction on a small, imbalanced dataset.

**Not intended** for: automated treatment decisions, whole-plant or field images, crops other than tomato, or any
setting where a wrong answer causes harm without a human check. Always confirm with an agricultural extension
service (in India, a Krishi Vigyan Kendra).

## Training and evaluation data

* Tomato-Village, Variant-a (Gehlot et al., 2023): 2,103 distinct photographs (4,525 files incl. the authors'
  augmented copies), one phone model (realme GT Master Edition), Jodhpur and Jaipur, Rajasthan.
* Leakage-free split by leaf group: 1,813 train+validation photographs (5-fold cross-validation for model choice and
  calibration) and 290 test photographs (265 in the focused scope), scored once.
* Severe imbalance: 1,024 Leaf Miner vs 9 Potassium photographs.

## Metrics

Accuracy and **macro-F1** (every class weighted equally) with 95% bootstrap confidence intervals over leaf groups;
per-class precision/recall/F1; expected calibration error; coverage and accuracy of answered photos at the
deployed threshold; flip rates under harmless image changes; out-of-distribution flag rates. Values: RESULTS.md §1–7.

## Quantitative results (test photos, scored once)

| Model | Accuracy (95% CI) | Macro-F1 | ECE | Answers → accuracy | Previous version (v1) |
|---|---:|---:|---:|---:|---|
| Focused (`upgrade_5class`, 265 photos) | 92.1% (88.4–95.3) | 0.844 | 0.055 | 81% → 98.1% | 90.9%, F1 0.835, 85% → 96.0% |
| Full (`upgrade_8class`, 290 photos) | 82.4% (77.7–87.0) | 0.559 | 0.076 | 88% → 87.4% | 84.8%, F1 0.586, 90% → 87.7% |

Paired bootstrap (v2 − v1): focused +1.1 accuracy points (95% CI −1.8 to +3.9), full −2.4 (−5.0 to 0.0) — neither
difference is reliable on this test set. The full model was chosen by a rule fixed on out-of-fold data before the
test set was scored; it is reported as chosen.
Unusual-image check: 100% of held-out non-leaf probe images flagged; 7.9% (focused) and 4.8% (full) of normal test
photos flagged; only 4% of photos of the three conditions outside the focused scope are flagged (20% receive no
diagnosis once “Uncertain” is included).

## Factors and known failure modes

* **Rare classes**: Potassium (2 test photos) and Nitrogen (7) dominate the uncertainty of macro-F1 on the full task.
* **Look-alikes**: Late Blight vs Spotted Wilt Virus, Leaf Miner vs Healthy at 256×256 (thin mines are a few pixels).
* **Capture conditions**: trained on leaves on paper under daylight; backgrounds, lighting and cameras outside this
  setting are out of distribution. The *Unusual image* check catches many such photos but not all.
* **Preprocessing**: photos are converted the way the dataset was made (whole frame to 256×256, Display P3,
  JPEG q90). Very wide or tall photos are centre-cropped to 4:3 / 3:4 first; keep the leaf centred.

## Ethical considerations

* Images are of leaves only; no personal data is processed or stored by the API. Diagnosis history is kept only in
  the user's browser.
* Dataset licence: none declared; the images are not redistributed beyond 32 example photos shown with citation.
* Over-reliance risk: the interface always shows uncertainty, thresholds and a disclaimer, and refuses to diagnose
  low-confidence or unusual photos.

## Caveats and recommendations

* Results come from one dataset; re-validate on local photos before any real use.
* Thresholds were fitted on single fold models but are applied to their average; the measured test coverage is
  reported next to each threshold.
* Re-run `scripts/19_baseline_audit.py`-style checks if the upload pipeline changes: small preprocessing mismatches
  changed several percent of predictions in the original app.
