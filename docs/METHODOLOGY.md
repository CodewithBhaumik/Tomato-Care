# Methodology

How the data were checked, how the models were trained and chosen without touching the test set, and how every
reported number was computed. All numbers quoted from experiments are in [RESULTS.md](RESULTS.md), which is
generated from the result files; this document explains the procedure.

## 1. Dataset

**Tomato-Village, Variant-a** (Gehlot, Saxena & Gandhi, *Multimedia Systems*, 2023, doi:10.1007/s00530-023-01158-y):
single tomato leaves photographed on paper in Jodhpur and Jaipur (Rajasthan, India), eight classes — Early Blight,
Healthy, Late Blight, Leaf Miner, Magnesium / Nitrogen / Potassium Deficiency, Spotted Wilt Virus.

| Fact | Value | How it was established |
|---|---|---|
| Files | 4,525 JPEGs, all 256×256, none corrupt or byte-identical | `scripts/01_build_index.py` (full decode, MD5) |
| Distinct photographs | 2,103 | file names `IMGyyyymmddhhmmss[_n]`: 346 photos have 7 augmented copies |
| Re-shots of the same leaf | 340 near-duplicate pairs merged | ImageNet embeddings (cosine ≥ 0.871) and capture times ≤ 120 s, `scripts/02` |
| Leaf groups | 1,786 | union–find over copies and re-shots |
| Imbalance | 1,024 Leaf Miner vs 9 Potassium photographs (114:1) | |
| Licence | none declared | images are not redistributed; cite the paper |

## 2. A leakage-free split

The official split puts augmented copies of one photograph into train, validation and test: 61.2% of the official
test images have a sibling in the training folder (100% for the five augmented classes). A model can score highly
by recognising photographs it has already seen. We therefore split **whole leaf groups** with `StratifiedGroupKFold`
(the best-balanced of 200 seeded draws, chosen by label counts only): 1,502 / 311 / 290 original photographs for
train / validation / test, with no leaf group on two sides (`scripts/03_make_splits.py`, asserted in tests).
Trained and tested on the official folders, the original study's MobileNetV2 and ConvNeXt-Tiny reach 93.9% and 97.6%
accuracy; with the leakage-free split the same recipes reach 74.8% and 82.1% — the difference measures the inflation.

## 3. Preprocessing audit

The original web app gave different answers for photos that looked alike. Before training anything new we checked
whether a real phone photo reaches the model in the same form as a training photo. It did not.

**Shape.** Every dataset photo was taken with a *realme GT Master Edition* at 3468×4624 (72%, portrait) or 4624×3468
(EXIF of all 2,103 originals) and stored at 256×256. The authors resized the **whole** 3:4 frame into the square —
a non-uniform squash. Evidence: registering each original against the authors' rotated copies (SIFT + RANSAC affine)
gives singular-value ratios up to 1.79 ≈ (4/3)² at 90° rotations (and ≈ 1 for flips), exactly the signature of
`squash · rotate · squash⁻¹`. The original upload path instead **centre-cropped a square**: on a phone photo the
leaf appeared 25% narrower than in training and the top and bottom eighths were cut away.

**Colour.** The dataset files carry a **Display P3** ICC profile and P3-encoded pixel values. The original upload
path ignored colour profiles, so an sRGB photo (most web images and many phones) reached the model with different
values — and colour is how nutrient deficiencies show.

**Compression.** Dataset files are JPEG quality 90 with 4:2:0 chroma (estimated from the quantisation tables of
200 files); a downscaled phone photo is much cleaner, and an earlier version flipped borderline predictions after a
q95 re-encode.

**Fix** (`tomatoleaf.preprocessing.prepare_upload`): EXIF orientation → convert into Display P3 (sRGB, untagged and
other ICC profiles via LittleCMS) → centre-crop the aspect ratio into [3:4, 4:3] → resize the whole frame to 256×256 →
JPEG quality 90. Dataset images pass through bit-for-bit unchanged (tested on all 4,525 files), so training data and
earlier results are unaffected.

**Measured effect** (`scripts/19_baseline_audit.py`, validation photos only): each validation photo is turned back
into a full-frame phone photo (3:4 or 4:3 as its EXIF says) saved either with a P3 profile or as untagged sRGB, then
sent through the old and the new upload path. For the original deployed model, the share of predictions that change
falls from 2.1% to 0.0% (P3 photo) and from 3.5% to 1.4% (sRGB photo). Tables for all models: RESULTS.md §5.

## 4. Model selection protocol

The original study selected models on one validation split of 311 photos containing a single Potassium and six
Nitrogen photos: one photo moved validation macro-F1 by up to 0.125, so the choice was largely noise. The upgrade
uses **grouped, stratified 5-fold cross-validation over the 1,813 train+validation photos**
(`scripts/20_make_cv_folds.py`; every class in every fold; leaf groups never split):

1. Each recipe trains for a **fixed number of epochs** — no early stopping — so no epoch is chosen by looking at the
   held-out fold; held-out scores are logged for the training curves only.
2. Every photo receives an **out-of-fold (OOF) prediction** from a model that never saw its leaf group. Recipes,
   test-time augmentation and ensembles are ranked on these 1,813 predictions.
3. The **deployed model of a recipe is its 5 fold models** (a fold ensemble). Each fold model has genuinely unseen
   data — its own fold — so its temperature and out-of-distribution threshold are fitted on photos it never saw.
4. The selection rule was fixed in code before the test set was scored (`scripts/22_select_models.py`):
   *highest pooled OOF macro-F1 among candidates whose estimated CPU time for the whole fold ensemble is ≤ 1 s per
   photo; ties within 0.001 → higher OOF accuracy → lower latency.*
5. The 290 test photos were then scored **once** (`scripts/25_final_benchmark.py`).

Outcome worth knowing: for the full task, out-of-fold data ranked the two-recipe ensemble first (macro-F1 0.634) and
the robust recipe alone lower; on the test photos the robust recipe alone scored higher (0.598 vs 0.559) and the
original study's model too (0.586). Recipes this close do not keep their order between data sets of this size. The
pre-registered choice was kept — choosing again after seeing test results would make the test score optimistic —
and every model's test result is published (RESULTS.md §3).

Caveat: the same 290 test photos were used to report the original study's results, so they are not perfectly
fresh; no decision in the upgrade used them.

## 5. Models and training recipes

| Recipe | Backbone (pretraining) | Changes vs original | Run |
|---|---|---|---|
| `convnext_t_in1k_base` | ConvNeXt-Tiny (ImageNet-1k, torchvision) | none — the original recipe, as the reference | 5 folds |
| `convnext_t_in1k_robust` | ConvNeXt-Tiny (ImageNet-1k) | robust recipe | 5 folds |
| `dinov2_s_robust` | ViT-S/14 (DINOv2 self-supervised, LVD-142M) | backbone + robust recipe, 252 px, layer decay 0.8 | 5 folds |
| `convnext_t_in22k_base` | ConvNeXt-Tiny (ImageNet-22k → 1k, timm) | backbone only (ablation) | fold 0 screen |
| `convnext_t_in22k_robust` | ConvNeXt-Tiny (ImageNet-22k → 1k) | backbone + robust recipe | fold 0 screen |
| `convnext_t_in22k_llrd` | ConvNeXt-Tiny (ImageNet-22k → 1k) | first robust version: heavy augmentation, layer decay 0.8, stochastic depth — under-fitted (train accuracy 0.92 vs 0.99) and was stopped | fold 0 |
| `dinov2_b_linear` | ViT-B/14 (DINOv2), **frozen** | no fine-tuning: CLS + mean-patch features, class-balanced logistic regression, C by nested grouped CV | 5 folds |

The ImageNet-22k variants were screened on fold 0 first (to save GPU time) and only recipes that looked competitive
were run on all five folds; all screens are reported in RESULTS.md §4. The frozen-feature baseline reached
out-of-fold macro-F1 0.574 and accuracy 0.771, well below fine-tuning (≈ 0.61–0.62 and ≈ 0.85), so it was not
deployed.

Common to all: 256×256 input (252 for the ViT), class-weighted cross-entropy with label smoothing 0.1 (weights
N / (K·n_c), applied to the true class only), AdamW, a head-only warm-up stage, then all layers with linear warm-up
and cosine decay, bf16 autocast, fixed seeds.

The **robust recipe** adds: (a) all eight flips/90° rotations as training augmentation, so every test-time
augmentation view is in-distribution (the original recipe never rotated by 90°, yet averaged 90° views at test time);
(b) aspect-ratio jitter 3:4…4:3 (the dataset mixes photos squashed along either axis); (c) JPEG re-compression
(quality 50–95) and mild Gaussian blur, so compression and resampling differences between uploads and dataset files
matter less; (d) layer-wise learning-rate decay (lower layers move less, keeping generic pretrained features);
(e) an exponential moving average of the weights, evaluated at the end; (f) gradient clipping and stochastic depth.
Hue is never jittered: colour is the diagnostic signal for deficiencies.

**Focused task.** The five-class scope (Healthy, Leaf Miner, Magnesium, Nitrogen, Spotted Wilt) was fixed in the
original study on validation F1 (the four conditions recognised best, plus Healthy); it is kept for comparability.

## 6. Calibration and uncertainty

* **Temperature scaling**: one temperature per recipe and view setting, fitted on its pooled OOF logits (minimum
  negative log-likelihood); ensembles average the members' calibrated probabilities.
* **“Uncertain”**: the lowest calibrated-confidence threshold whose OOF accuracy on answered photos reaches the
  target (95% for the focused task, 90% for the full task) with at least 50% coverage.
* **“Unusual image”** (out-of-distribution): each fold model scores a photo by the cosine distance to the k-th
  nearest (k = 5) of its training photos in penultimate-feature space (kNN, Sun et al. 2022), or by the Mahalanobis
  distance to class means (Lee et al. 2018, Ledoit–Wolf covariance) — whichever separates OOD inputs better on OOF
  data. Each fold model's threshold keeps 95% of its own held-out photos; the ensemble flags a photo when the mean
  ratio score/threshold, rescaled on OOF data, exceeds 1. OOD inputs for choosing the score: training photos of the
  three out-of-scope conditions (focused task) and half of a fixed set of 30 non-leaf probe images (scikit-image
  photographs and synthetic images); the other half is used only for reporting.

## 7. Final evaluation

All models — classical ML, the original networks and the upgrade — are scored by the **same predictor code that the
API runs** (`backend/services/predictor.py`), in float32, on identical test photos (CPU/GPU parity checked). Reported:
accuracy, macro precision/recall/F1, weighted F1; 95% confidence intervals from 2,000 bootstrap resamples of whole
leaf groups; differences between models from a **paired** bootstrap on the same resamples; per-class metrics;
confusion matrices; calibration (ECE with 15 bins, NLL, Brier); selective accuracy at each model's threshold;
robustness to harmless changes and to simulated phone uploads; CPU latency (median, 8 threads).

## 8. Limitations

* One phone, two districts, single leaves on paper: performance on field photos, other cameras or regions is unknown.
* Potassium Deficiency has 9 photographs in total (2 in the test set) and Nitrogen Deficiency 45: their per-class
  scores are very uncertain, and macro-F1 on the full task inherits that uncertainty.
* 290 test photos: differences of a few points are within noise — read the confidence intervals and paired tests.
* Thresholds come from OOF predictions of single fold models but are applied to the 5-model average, which is
  typically less confident; the realised coverage on new photos can differ (measured on test in RESULTS.md).
* The OOD check was evaluated on out-of-scope leaf conditions and generic images, not on other crops or field scenes.
* Grad-CAM shows where a network looked, not why a leaf is diseased.

## 9. Reproducing

```bash
python scripts/01_build_index.py && python scripts/02_near_duplicates.py && python scripts/03_make_splits.py
python scripts/19_baseline_audit.py                     # audit of the original models (validation only)
python scripts/20_make_cv_folds.py
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in1k_base      # all 5 folds; likewise for
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in1k_robust    # 8class dinov2_s_robust and 5class
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in22k_base --folds 0   # screening runs: fold 0 only
python scripts/21b_frozen_features.py --task 8class                            # frozen DINOv2 ViT-B baseline
python scripts/22_select_models.py --task 8class && python scripts/22_select_models.py --task 5class
python scripts/23_build_deployment.py
python scripts/24_cpu_inference.py                      # optional: CPU inference options
python scripts/25_final_benchmark.py                    # the test set, once
python scripts/26_package_release.py && python scripts/27_export_platform_data.py && python scripts/29_results_report.py
python scripts/28_e2e_check.py                          # React platform in a browser (after npm run build)
streamlit run streamlit_app.py --server.port 8510 & python scripts/30_streamlit_check.py   # Streamlit app in a browser
```

## 10. Serving

The REST API exports each network once to ONNX Runtime (about 1.6–2× faster on a CPU, outputs within 1e-6 of
PyTorch). The Streamlit app runs on small shared servers in **low-memory mode** (`TOMATO_LOWMEM=1`): all fold
networks of one architecture share a single float32 network and copy their memory-mapped float16 weights in just
before running — bit-identical predictions, 0.9 GB instead of 2.4 GB of private memory for the two version-2 models
(`tests/test_lowmem.py`).

Hardware used: one RTX 4050 Laptop GPU (6 GB); software versions in `requirements*.txt` and
`experiments/platform.json → environment`.
