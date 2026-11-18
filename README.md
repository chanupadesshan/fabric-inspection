# Fabric Inspection: Classical Anomaly Detection and Colormap Research

Layer 1 of a two-layer fabric inspection system on the AITEX dataset. First built **without ML/DL**, using classical image processing and colormap/heatmap visualization (notebooks 01-21); then compared with models trained on good fabric only (notebooks 22-23), on the same test tiles and metrics.

## Getting started

```
git clone <repository url>
cd <repository folder>
python -m venv .venv
.venv\Scripts\activate                  # Windows (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Put the AITEX images in `data/raw/` (see `data/README.md`; they are not part of the repository).
2. `python src/preprocess.py` makes the tiles in `data/processed/`.
3. Open the notebooks in `experiments/` (`jupyter notebook experiments`) or start the app: `streamlit run app/app.py`.

The experiment notebooks are stored **without outputs** (pictures are drawn again when you run them; scores and
thresholds are in `results/`). Before committing notebooks you have run, strip their outputs:
`python tools/strip_notebook_outputs.py`.

Status: notebooks 01-26 have been run. `27_rd4ad` and `28_efficientad` are written but not yet run or tuned.

## 1. Goal

| Layer | Question | Status |
|---|---|---|
| **Layer 1: Anomaly** | Is this fabric damaged or not? Where? | In progress (this README) |
| **Layer 2: Damage type** | Which exact damage is it (hole, broken yarn, stain, ...)? | Later |

The aim of this stage is to find out how far classical methods can go on Layer 1 before moving to ML. The measured gap between the best classical result and an ML baseline is the evidence for (or against) going to ML.

## 2. Key idea: detection vs. visualization

```
tile -> enhance -> anomaly map -> tile score -> threshold -> OK / DAMAGED
                        |
                        +-> threshold + min-area filter -> damage mask (where; IoU only)
                        +-> colormap -> heatmap overlay (viewing only)
```

- The **anomaly map** (a grayscale map of "how different from normal fabric") makes the OK / DAMAGED decision through the **tile score** (section 7).
- The **damage mask** (pixels above the threshold, tiny spots removed) only shows *where* the damage is and is used for IoU. Keep its minimum area **<= 4 px**: the smallest labelled defects are only a few pixels, so a larger filter deletes real defects.
- The **colormap** (JET, TURBO, INFERNO, ...) only changes how the map looks. Two colormaps on the same map give the same detection result.
- Decisions use the anomaly values, never the colours. Heatmaps use a **display range tied to the threshold** (`F.display(amap, vmax=2 x the tile's pixel threshold)`, so mid-scale = "would be flagged"), not per-image min-max, so good fabric does not show fake hot spots.

## 3. Dataset: AITEX

- Grayscale woven fabric strips, about 4096 x 256 pixels, several fabric types.
- Defect-free images, plus defective images with ground-truth masks.
- File names follow `imageNumber_defectCode_fabricCode` (for example `0044_019_04`): the middle number is the defect type (`000` = no defect), the last number is the fabric type. Verify the code meanings against the dataset documentation.
- Some defective images have several masks (`_mask1`, `_mask2`): `0044_019_04` and `0097_030_03`. `src/preprocess.py` merges them into one mask (pixel-wise max).
- One defective image has no mask in the dataset (`0100_025_08`) and is skipped.
- One defective image has an **empty mask** (`0106_010_03`, all zeros): labelled defective but nothing marked, so it is skipped (it is neither a defect example nor a trusted clean one).
- Masks have soft (anti-aliased) edges with grey values, not pure 0/255, so they are thresholded at 127.
- One defective strip is narrower (`0094_027_05`, 3796 px wide), so it gives 14 tiles instead of 16; the dropped 212 px strip has no defect pixels.
- Fabric 01 has 21 defect-free strips instead of 20: the extra one is `0018_00_01.png` (two-digit defect code). It is a different image from `0018_000_01.png` (checked pixel by pixel), not a duplicate, so it is kept. There are no duplicate raw files (`src/preprocess.py` checks this on every run).

### Folder structure

```
fabric inspection/
  README.md
  app/
    app.py                # Streamlit app: upload photos -> damage location, size input, four-point score
  src/
    preprocess.py         # raw strips -> 256 x 256 tiles (section 4)
    methods/
      filters.py          # ALL filters and methods (numpy + OpenCV only)
      evaluate.py         # shared evaluation (section 7): split, threshold, metrics, plots, saving
      models/             # learned models (needs torch + torchvision)
        __init__.py       # import models as M -> M.fit_<model>, M.<model>_map, M.anomaly_map
        backbone.py       # shared: frozen pretrained network and feature extraction
        patchcore.py      # PatchCore: memory bank of good patches
        padim.py          # PaDiM: Gaussian per patch position
        rd4ad.py          # RD4AD: reverse distillation (trains a bottleneck + decoder)
        efficientad.py    # EfficientAD: student-teacher + autoencoder (trains both)
        _train.py         # shared helpers for the two that train
  experiments/            # one notebook per experiment (table below)
    01_background_subtraction.ipynb ... 21_combined.ipynb    # classical
    22_patchcore.ipynb, 23_padim.ipynb                        # models
    24_ensemble.ipynb                                         # classical + model together
    25_real_photos.ipynb                                      # the models on real photos (data/real)
    26_colormap.ipynb                                         # colormap comparison (section 6)
    27_rd4ad.ipynb, 28_efficientad.ipynb                      # models that train (not yet run)
  data/
    raw/
      Defect_images/
      Mask_images/
      NODefect_images/
    processed/            # created by preprocess.py
      good/                       # tiles from defect-free images
      edge/                       # tiles with background or the fabric edge (not used)
      defect/                     # tiles whose mask has defect pixels
      mask/                       # binary mask per defect tile
      defect_image_clean_tile/    # clean tiles from defective images (hard negatives)
      tiles.csv                   # one row per tile: source, fabric, defect code, folder, defect px
    real/                 # real photos (not AITEX)
  results/                # written by the notebooks
    <method>/
      scores_fabricXX.csv         # per-tile scores and predictions
      split_fabricXX.json         # good_cal / good_test source strips, threshold, settings
    colormap/             # pictures from 26_colormap
    real_photos/          # pictures from 25_real_photos
  docs/
    README_old.md         # earlier version of this file
```

Run from the project folder:

```
python .\src\preprocess.py          # once: makes data/processed
jupyter notebook experiments        # the experiments (each notebook adds ../src/methods to the path)
streamlit run app\app.py            # the app (needs streamlit; first start trains PatchCore, a few minutes)
```

### Code layout rule

- **`src/methods/filters.py` holds every filter and method.** No filter code is written inside a notebook.
  Sections: 1 enhancement, 2 intensity/statistics, 3 morphology, 4 frequency, 5 texture,
  6 edge/line, 7 reference-based, 8 classical learning, 9 helpers.
- Every method takes a grayscale tile and returns a float32 anomaly map (higher = more abnormal).
  `filters.METHODS` lists the tile-only ones; `filters.BINARIZERS` lists Otsu and adaptive threshold (section 5).
- **`src/methods/models/` holds the learned models**, one file each (`patchcore.py`, `padim.py`) plus the shared `backbone.py`. They follow the same contract:
  a fitted model turns a tile into an anomaly map, so they are evaluated exactly like the classical methods.
  This is the only folder that needs PyTorch.
- **`src/methods/evaluate.py` holds the evaluation**, the same for every experiment (section 7). It contains
  no filters; it only scores a method. Methods that need good fabric first (texture reference, golden
  sample, learned models, fusion) pass a `fit` function that receives the `good_cal` tiles only.
- **One notebook = one experiment.** A notebook imports `filters.py` (`import filters as F`) and
  `evaluate.py` (`import evaluate as E`), sets the method and its parameters in the first code cell,
  runs it and records the result. If an experiment needs a new filter, add it to `filters.py` first,
  then call it from the notebook.
- Notebooks are numbered in the order they are run: `NN_<experiment>.ipynb`.

| Notebook | Experiment |
|---|---|
| `01_background_subtraction.ipynb` | Baseline: blur = normal fabric, anomaly = difference |
| `02_fft_notch.ipynb` | Remove the weave's spectrum peaks, anomaly = what is left |
| `03_gabor_bank.ipynb` | Gabor filter energy vs. its local average |
| `04_tophat_blackhat.ipynb` | Small bright / dark details (morphology) |
| `05_local_zscore.ipynb` | Pixel vs. local mean and std |
| `06_sobel.ipynb` | Gradient energy vs. its local average |
| `07_opening_closing.ipynb` | Opening / closing residual (morphology) |
| `08_wavelet.ipynb` | Haar wavelet detail energy |
| `09_periodicity.ipynb` | Pixel vs. pixel one weave period away |
| `10_lbp.ipynb` | LBP patch features vs. good-fabric reference (`fit`) |
| `11_glcm.ipynb` | GLCM patch features vs. good-fabric reference (`fit`) |
| `12_hog.ipynb` | HOG patch features vs. good-fabric reference (`fit`) |
| `13_patch_features.ipynb` | All patch features combined vs. reference (`fit`) |
| `14_canny.ipynb` | Canny edge map |
| `15_hough.ipynb` | Long lines (Hough) |
| `16_lsd.ipynb` | Long line segments (LSD) |
| `17_golden_subtraction.ipynb` | Difference to a golden sample from good_cal (`fit`) |
| `18_ssim.ipynb` | 1 − SSIM against the golden sample (`fit`) |
| `19_one_class_svm.ipynb` | One-class SVM on patch features, trained on good_cal (`fit`) |
| `20_isolation_forest.ipynb` | Isolation Forest on patch features, trained on good_cal (`fit`) |
| `21_combined.ipynb` | Fusion of the best 2-3 methods (edit `FUSE` after comparing) |
| `22_patchcore.ipynb` | Model: memory bank of good-patch features from a pretrained network; anomaly = distance to the nearest (`fit`) |
| `23_padim.ipynb` | Model: Gaussian fitted to good-patch features from a pretrained network; anomaly = Mahalanobis distance (`fit`) |
| `24_ensemble.ipynb` | Ensemble: background subtraction + PatchCore judge the same tile, one rule combines them (edit `MEMBERS`, `RULE`) |
| `25_real_photos.ipynb` | The models and the ensemble on real photos (`data/real`): five-column picture per photo; optional calibration on good real photos (`data/real_good`) |
| `26_colormap.ipynb` | Same fixed-range anomaly map in JET, TURBO, INFERNO, MAGMA, VIRIDIS, HOT: grid and overlay (section 6) |
| `27_rd4ad.ipynb` | Model: reverse distillation. A decoder is trained on good tiles to rebuild a frozen teacher's features; anomaly = rebuild error (`fit`, trains: minutes). Code written, not yet run |
| `28_efficientad.ipynb` | Model: EfficientAD. Student and autoencoder trained on good tiles against a teacher; anomaly = their disagreement (`fit`, trains: many minutes). Code written, not yet run |

Every notebook has the same cells: settings -> one fabric (summary, saved results) -> score histogram + ROC
-> recall per defect type -> failure cases -> example tiles -> one example per defect type (all 12 types,
each judged with its own fabric's threshold) -> all fabrics.
Slow ones (about 1-3 min per fabric): `11_glcm`, `13_patch_features`, `19_one_class_svm`, `20_isolation_forest`, and `22_patchcore` on a CPU.
The model notebooks have one more section at the end: an optional hyperparameter comparison (`RUN_TUNING`).

Adding an experiment: copy any notebook inside `experiments/` to `NN_<experiment>.ipynb`, change `METHOD_NAME`, the method
settings and `method` (or `fit`) in the first code cell, and add a row to the table above.

## 4. Preprocessing

`src/preprocess.py` cuts each strip into non-overlapping 256 x 256 tiles (16 per strip) and saves them with the name `<source>_tXX.png`. Masks are tiled the same way. Saved masks are binary (0 / 255).

Tile rules:
- **defect**: the tile's mask has any pixel above 127 (this ignores the soft mask edges).
- **edge**: the tile contains background or the fabric edge (not used for evaluation).
- **defect_image_clean_tile**: a tile from a defective image with no defect pixel **and** more than 8 px (`CLEAN_MARGIN`) away from the mask. These are the hardest negatives (same strip, same lighting).
- **near_defect**: no defect pixel, but within 8 px of one. Not trusted as clean, not saved (listed in `tiles.csv` only).

The script clears the output folders before writing, so re-running never leaves stale tiles behind. Paths are resolved from the script's own folder. On every run it also prints duplicate raw files and a per-fabric count of tiles and source strips per folder: check these against the table below.

Run (works from any folder):

```
python .\src\preprocess.py
```

Result for this download: `good` 1878, `edge` 640, `defect` 184, `mask` 184, `defect_image_clean_tile` 1208, `near_defect` 8 (not saved).

Tiles per fabric type (fabric code = third part of the file name, `name.split("_")[2]`; the second part is the defect code). Tiles / source strips:

| Fabric | good | edge | defect | clean tiles from defect images | near_defect |
|---|---|---|---|---|---|
| 00 | 220 / 20 | 125 | 9 / 7 | 76 / 7 | 2 |
| 01 | 294 / 21 | 58 | 28 / 12 | 146 / 12 | 2 |
| 02 | 280 / 20 | 100 | 38 / 38 | 507 / 38 | 3 |
| 03 | 220 / 20 | 234 | 89 / 30 | 256 / 28 | 1 |
| 04 | 276 / 20 | 63 | 12 / 10 | 129 / 10 | 0 |
| 05 | 300 / 20 | 20 | 1 / 1 | 13 / 1 | 0 |
| 06 | 288 / 20 | 40 | 7 / 6 | 81 / 6 | 0 |

Fabric 02 and 03 have the most defects and are the best first test. Fabric 05 has only one defect tile, too few to measure recall.

Notes:
- Tiles do not overlap, so a defect on a tile border can be cut in half (some defect tiles hold only a few defect pixels, e.g. `0066_023_01_t09` with 4). Overlap (stride 128) is an option if border misses appear.
- Many real defects are tiny (for example defect code 019: 8-20 pixels), so do not drop defect tiles by size without checking.
- Class imbalance: 184 defect tiles vs. about 3100 defect-free tiles.
- Always **split by source image, not by tile**. Tiles from the same strip look alike, and mixing them across sets inflates results.
- Use **one threshold per fabric type**.

## 5. Methods to evaluate

Contract: each method is a function that takes a grayscale tile and returns a float32 anomaly map (higher = more abnormal), so all are evaluated by the same rules.

| Group | Methods |
|---|---|
| Image enhancement | Gaussian/median filter, flat-field correction, CLAHE (applied to each tile inside a method via `steps=`; not part of `src/preprocess.py`). Default `DEFAULT_STEPS = ("gaussian",)`, chosen for fabric 02 |
| Intensity / statistics | Background subtraction, local z-score (3-sigma) |
| Morphology | Top-hat / black-hat, opening / closing |
| Frequency | FFT notch filter, Gabor bank, wavelet, periodicity analysis |
| Texture | LBP, GLCM, HOG, patch-wise feature distance |
| Edge / line | Sobel, Canny, Hough, LSD (line defects) |
| Reference-based | Golden-sample subtraction, SSIM (optional, needs alignment) |
| Classical learning | One-class SVM, Isolation Forest on texture features (last step) |
| Decision stage (not detectors) | Otsu, adaptive threshold (`filters.BINARIZERS`) |
| Color | Not used: AITEX images are grayscale |

Notes on the contract:
- **Otsu and adaptive threshold are binarizers, not anomaly scorers.** They always split an image into two classes, so on good fabric they still mark pixels. Use them on an anomaly map at the decision stage, not as detectors.
- **LBP / GLCM / HOG** are descriptors. The texture methods return the **distance of each patch's descriptor to the good-fabric reference** (fitted on good tiles with `fit_patch_reference`), not the raw descriptor.
- **FFT notch**: the tile is mirror-padded before filtering and the weave peaks are found on a Hann-windowed spectrum, so the tile border does not create a leakage cross.
- **Combined map** (`fit_fusion` + `fuse`): normalise each method's map with **fixed `good_cal` statistics per fabric** (median and 99.9th percentile of cal pixels), then take the pixel-wise max. Never per-tile min-max: it amplifies noise on good tiles (same principle as the fixed colormap range).

Order: background subtraction first (baseline), then FFT, Gabor, top-hat, local z-score, Sobel, then LBP/GLCM. Add a method only when the previous one fails on a specific case.

## 6. Colormap comparison

Compare how the same fixed-range anomaly map looks under different colormaps (`cv2.COLORMAP_*`). Display range: `vmax = 2 x` the cal-derived pixel threshold, so the middle of the colormap means "would be flagged" and the heatmap agrees with the decision.

| Colormap | Notes |
|---|---|
| JET | Familiar, but adds fake bands and edges |
| TURBO | Smoother replacement for JET |
| INFERNO / MAGMA / VIRIDIS | Perceptually smooth, honest brightness; good for reports |
| HOT | Simple black-red-white, shows peaks |

Output: a side-by-side grid image per example tile (good tile and defect tile), plus an overlay on the original image (`cv2.addWeighted`).

## 7. Evaluation protocol

Per fabric type:

| Set | Use |
|---|---|
| `good_cal` | Set the threshold. Half of the good source strips (fixed seed) |
| `good_test` | Measure false alarm rate on unseen good strips |
| Hard negatives (`defect_image_clean_tile`) | Measure false alarm rate on clean tiles from defective strips; test only |
| `defect` tiles | Measure recall; test only, never tune on them |

All tiles from defective strips (defect tiles and their clean tiles) stay on the test side.

- **Tile score** (`F.tile_score`): Gaussian-smooth the anomaly map (sigma = 2 px), then take the **mean of the top 25 values** (about the smallest real defect). A 99.9th percentile is the ~66th brightest pixel of a 256 x 256 tile and cannot see a 4-20 px defect.
- **Normalised score** (`normalize=True`, notebook setting `NORMALIZE`): the score is divided by the **tile's own typical anomaly level** (median of the smoothed map), so it reads "the strongest spot is N times this tile's normal texture response". Why: whole strips differ in contrast / lighting / weave tightness, which scales the whole anomaly map. With the raw score, clean tiles of the defective strips of fabric 02 scored up to 21 against a threshold of 16.4 set on good strips (40 % false alarms on hard negatives); normalised, 3.4 %. It needs only the tile itself, so a strip that is damaged along most of its length is still caught (a score relative to the strip's median tile was tried and hid those defects on fabrics 01 and 03).
  - Off for 0 / 255 maps (`14_canny`, `15_hough`, `16_lsd`: the typical level is often 0) and for `19_one_class_svm` (signed score).
  - `21_combined`: `F.fuse` divides each map by the tile's own level before combining, so the fused map is scored without a second normalisation.
- **Threshold** (`F.robust_threshold`), from `good_cal` only: `median + 4 x 1.4826 x MAD` of the cal tile scores. Not a multiplicative margin (that fails when good tiles score ~0, e.g. top-hat, and one odd cal tile can push it up). The damage mask uses the same decision per pixel: smoothed map > threshold x the tile's level.
- **Metrics**:
  - Tile level: false alarm rate on `good_test` and on hard negatives, recall, recall split by defect size (< 50 px vs >= 50 px of mask area), AUROC (threshold-free), IoU against the mask, ms per tile.
  - Strip level: a strip is flagged if any tile is flagged. Strip false alarm rate on `good_test` strips, strip recall on defective strips. With 16 tiles per strip, 1 % tile false alarms is about 15 % strip false alarms, so tile false alarms must be pushed well below 1 %.
  - The ROC / false alarm vs recall curve is computed on test tiles for analysis only; the operating point comes from `good_cal`.
- Priority: high recall first, then reduce false alarms.
- **Models (notebooks 22-23)** use the same split and the same test tiles. The `good_cal` strips are split once more (`FIT_FRACTION = 0.7`): 70 % of them train the model, the other 30 % set the threshold. A model scores the tiles it was fitted on as "too normal", so the same tiles cannot do both.
- **Ensemble (notebook 24)**: several members (classical methods and/or models) judge the same tile. Each member's tile score is standardised with its own `good_cal` statistics (`z` = robust deviations above good fabric; a member alone flags at `z > THR_K`), then one rule combines them: `max` = damaged if ANY member flags, `mean` = average `z` (threshold from the `good_cal` strips no member was fitted on), `min` = damaged only if ALL flag. Same split and test tiles as everything else; one run reports every member alone and every rule.
- **Hyperparameter tuning (models)**: only on one development fabric (fabric 02), with `E.tune`. Choosing a setting from that table uses fabric 02's defect tiles, so fabric 02's numbers become optimistic; the chosen setting is judged on the other fabrics. Never tune per fabric on its own test tiles.
- Reproducibility: every run writes `results/<method>/scores_fabricXX.csv` (per tile: peak, level, score, prediction) and `split_fabricXX.json` (cal/test source strips, threshold, settings). Threshold sweeps, fusion and the table below are recomputed from these files.

## 8. Results

Per fabric type, from `results/<method>/`. FAR = false alarm rate, hn = hard negatives.

Background subtraction baseline (`DEFAULT_STEPS`, `bg_sigma` 15, threshold k = 4, normalised score):

| Fabric | FAR test | FAR hn | Recall | Small / large | Strip FAR | Strip recall | AUROC | IoU | ms/tile |
|---|---|---|---|---|---|---|---|---|---|
| 00 | 1.8 % | 34.2 % | 77.8 % | 100 % / 75.0 % | 10.0 % | 85.7 % | 0.922 | 0.18 | 5.1 |
| 01 | 0.6 % | 8.2 % | 92.9 % | 50.0 % / 96.2 % | 9.1 % | 91.7 % | 0.981 | 0.14 | 5.5 |
| 02 | 1.4 % | 3.4 % | 89.5 % | 82.6 % / 100 % | 20.0 % | 89.5 % | 0.984 | 0.44 | 5.3 |
| 03 | 0.0 % | 5.9 % | 53.9 % | 69.2 % / 51.3 % | 0.0 % | 83.3 % | 0.817 | 0.09 | 5.4 |
| 04 | 5.8 % | 9.3 % | 83.3 % | 100 % / 71.4 % | 80.0 % | 90.0 % | 0.941 | 0.42 | 6.1 |
| 05 | 4.7 % | 0.0 % | 100 % | n/a / 100 % | 60.0 % | 100 % | 0.988 | 0.24 | 5.2 |
| 06 | 14.0 % | 18.5 % | 71.4 % | 60.0 % / 100 % | 100 % | 83.3 % | 0.936 | 0.19 | 5.4 |

All fabrics together: recall 71.2 %, false alarms 6.4 % of all negative tiles (raw score before normalisation: 64.1 % and 13.3 %).

Comparison across methods (fill in per fabric):

| Fabric | Method | FAR test | FAR hn | Recall | Strip FAR | AUROC | ms/tile | Notes |
|---|---|---|---|---|---|---|---|---|
| | Background subtraction | | | | | | | baseline (table above) |
| | FFT notch | | | | | | | |
| | Gabor | | | | | | | |
| | Top-hat | | | | | | | |
| | Combined (best 2-3) | | | | | | | `fuse`: pixel-wise max of cal-normalised maps |

## 9. Next steps

1. ~~Fix preprocessing (multi-mask images, mask threshold, stale outputs).~~ Done.
2. ~~Count tiles per fabric code.~~ Done (table in section 4).
3. ~~Audit tile counts and duplicates.~~ Done: no duplicates; empty-mask image `0106_010_03` skipped; clean tiles need an 8 px margin from the mask.
4. ~~Fix tile score (top-k after smoothing) and threshold (median + k x MAD); add hard negatives, strip-level and size-split metrics, CSV + split manifest.~~ Done.
5. ~~Run the background subtraction baseline.~~ Done (section 8).
   - ~~Hard negatives on fabric 02 gave 40 % false alarms.~~ Cause: strip-to-strip contrast differences. Fixed with the normalised score (section 7).
   - Open: false alarms are still high on hard negatives of fabric 00 (34 %) and on fabric 06 (14 % test, 18.5 % hard negatives); strip false alarms are high on fabrics 04, 05, 06.
   - Open: the damage mask marks only the peaks of a defect (low IoU), and long faint lines score low (the score uses the 25 strongest pixels only).
6. Check mask overlays on a few defect tiles to confirm tile/mask alignment (mask and image sizes already verified to match).
7. Add FFT, Gabor and the other methods and compare in the table above.
8. Combine the best methods into one anomaly map (`fuse`) and produce the colormap figures.
9. Compare the best classical result with models trained on good tiles only (same splits, same metrics): PatchCore and PaDiM are in place (`22_patchcore`, `23_padim`); run them on all fabrics and fill section 8. Then RD4AD / EfficientAD if needed, then Layer 2.
   - Open: the good strips used for training look slightly different from the strips that contain defects (the same strip-to-strip difference that biased the classical scores). A model trained on the good strips only scores whole defective strips as "different". Option to test: also train on clean tiles from some defective strips, and keep those strips out of the test set.

## 10. Requirements

- Python 3, `opencv-python`, `numpy` for `src/preprocess.py` and `filters.py`. Everything in `filters.py` (LBP, GLCM, HOG, wavelet, One-class SVM, Isolation Forest) is written in numpy + OpenCV, so `scikit-image`, `scipy`, `PyWavelets` and `scikit-learn` are not needed.
- Notebooks also need `matplotlib` and `ipykernel`.
- Everything is listed in `requirements.txt` (`pip install -r requirements.txt`); the app also needs `streamlit` and `pandas`.
- Model notebooks (22-23) also need `torch` and `torchvision` (`pip install torch torchvision`; from PyPI this is the CPU build on Windows, a GPU build comes from the PyTorch download page). The first run downloads the pretrained network weights (`IMAGENET1K_V1`) from download.pytorch.org; on a machine that cannot reach it, pass `weights=<path to a saved state_dict>` in `PARAMS`.
- Versions in the project `.venv`: Python 3.14.6, opencv-python 5.0.0.93, numpy 2.5.4, matplotlib 3.11.2, ipykernel 7.4.0.
- Dataset: AITEX (download separately and extract into `data/raw/`).
