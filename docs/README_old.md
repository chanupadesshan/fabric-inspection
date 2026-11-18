# Fabric Inspection: Classical Anomaly Detection and Colormap Research

Layer 1 of a two-layer fabric inspection system, built **without ML/DL**, using classical image processing and colormap/heatmap visualization on the AITEX dataset.

## 1. Goal

| Layer | Question | Status |
|---|---|---|
| **Layer 1: Anomaly** | Is this fabric damaged or not? Where? | In progress (this README) |
| **Layer 2: Damage type** | Which exact damage is it (hole, broken yarn, stain, ...)? | Later |

The aim of this stage is to find out how far classical methods can go on Layer 1 before moving to ML. The measured gap between the best classical result and an ML baseline is the evidence for (or against) going to ML.

## 2. Key idea: detection vs. visualization

```
tile -> enhance -> anomaly map -> threshold + area filter -> OK / DAMAGED
                            |
                            +-> colormap -> heatmap overlay (viewing only)
```

- The **anomaly map** (a grayscale map of "how different from normal fabric") makes the OK / DAMAGED decision.
- The **colormap** (JET, TURBO, INFERNO, ...) only changes how the map looks. Two colormaps on the same map give the same detection result.
- Decisions use the raw anomaly values. Heatmaps use a **fixed display range** (for example `clip(anomaly / 60, 0, 1)`), not per-image min-max, so good fabric does not show fake hot spots.

## 3. Dataset: AITEX

- Grayscale woven fabric strips, about 4096 x 256 pixels, several fabric types.
- Defect-free images, plus defective images with ground-truth masks.
- File names follow `imageNumber_defectCode_fabricCode` (for example `0044_019_04`): the middle number is the defect type (`000` = no defect), the last number is the fabric type. Verify the code meanings against the dataset documentation.
- Some defective images have several masks (`_mask1`, `_mask2`): `0044_019_04` and `0097_030_03`. `preprocess.py` merges them into one mask (pixel-wise max).
- One defective image has no mask in the dataset (`0100_025_08`) and is skipped.
- Masks have soft (anti-aliased) edges with grey values, not pure 0/255, so they are thresholded at 127.
- One defective strip is narrower (`0094_027_05`, 3796 px wide), so it gives 14 tiles instead of 16; the dropped 212 px strip has no defect pixels.

### Folder structure

```
fabric inspection/
  data/
    raw/
      Defect_images/
      Mask_images/
      NODefect_images/
    processed/            # created by preprocess.py
      good/                       # tiles from defect-free images
      defect/                     # tiles whose mask has defect pixels
      mask/                       # binary mask per defect tile
      defect_image_clean_tile/    # defect-free tiles from defective images
  methods/
    filters.py                    # ALL filters and methods of section 5 (numpy + OpenCV only)
  Notebooks/                      # one notebook per method, each tests functions from filters.py
    01_enhance.ipynb              # which enhancement filter makes defects stand out most?
    02_background_subtraction.ipynb  # baseline detector: false alarms, recall (+ evaluation helpers)
  preprocess.py
  README.md

`filters.py` sections: 1 enhancement, 2 intensity/statistics, 3 morphology, 4 frequency, 5 texture,
6 edge/line, 7 reference-based, 8 classical learning, 9 helpers. Every method takes a grayscale tile
and returns a float32 anomaly map (higher = more abnormal). `filters.METHODS` lists the tile-only ones.

Adding a method notebook: copy `02_background_subtraction.ipynb` to `NN_<name>.ipynb` and change
`method` to the function from `filters.py`, e.g. `F.fft_notch`.
```

## 4. Preprocessing

`preprocess.py` cuts each strip into non-overlapping 256 x 256 tiles (16 per strip) and saves them with the name `<source>_tXX.png`. Masks are tiled the same way. A tile counts as a defect tile if its mask has any pixel above 127 (this ignores the soft mask edges). Saved masks are binary (0 / 255).

The script clears the four output folders before writing, so re-running never leaves stale tiles behind. Paths are resolved from the script's own folder.

Run (works from any folder):

```
python .\preprocess.py
```

Result for this download: `good` 2256, `defect` 184, `mask` 184, `defect_image_clean_tile` 1494.

Tiles per fabric type (fabric code = third part of the file name, `name.split("_")[2]`; the second part is the defect code):

| Fabric | good tiles | defect tiles | defect source images | clean tiles from defect images |
|---|---|---|---|---|
| 00 | 320 | 9 | 7 | 103 |
| 01 | 336 | 28 | 12 | 164 |
| 02 | 320 | 38 | 38 | 570 |
| 03 | 320 | 89 | 30 | 407 |
| 04 | 320 | 12 | 10 | 148 |
| 05 | 320 | 1 | 1 | 13 |
| 06 | 320 | 7 | 6 | 89 |

Fabric 02 and 03 have the most defects and are the best first test. Fabric 05 has only one defect tile, too few to measure recall.

Notes:
- Tiles do not overlap, so a defect on a tile border can be cut in half (some defect tiles hold only a few defect pixels, e.g. `0066_023_01_t09` with 4). Overlap (stride 128) is an option if border misses appear.
- Many real defects are tiny (for example defect code 019: 8-20 pixels), so do not drop defect tiles by size without checking.
- Class imbalance: 184 defect tiles vs. about 3750 defect-free tiles.
- Always **split by source image, not by tile**. Tiles from the same strip look alike, and mixing them across sets inflates results.
- Use **one threshold per fabric type**.

## 5. Methods to evaluate

Each method is a function that takes a float32 grayscale tile and returns an anomaly map, so all are evaluated by the same rules.

| Group | Methods |
|---|---|
| Image enhancement | Gaussian/median filter, flat-field correction, CLAHE (applied to each tile inside a method; not part of `preprocess.py`) |
| Intensity / statistics | Background subtraction, local z-score (3-sigma), adaptive threshold, Otsu |
| Morphology | Top-hat / black-hat, opening / closing |
| Frequency | FFT notch filter, Gabor bank, wavelet, periodicity analysis |
| Texture | LBP, GLCM, HOG, patch-wise feature distance |
| Edge / line | Sobel, Canny, Hough, LSD (line defects) |
| Reference-based | Golden-sample subtraction, SSIM (optional, needs alignment) |
| Classical learning | One-class SVM, Isolation Forest on texture features (last step) |
| Color | Not used: AITEX images are grayscale |

Order: background subtraction first (baseline), then FFT, Gabor, top-hat, local z-score, Sobel, then LBP/GLCM. Add a method only when the previous one fails on a specific case.

## 6. Colormap comparison

Compare how the same fixed-range anomaly map looks under different colormaps (`cv2.COLORMAP_*`):

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
| `good_cal` | Set the threshold (for example 1.1 x the highest tile score on these) |
| `good_test` | Measure false alarm rate on unseen good tiles |
| `defect` tiles | Measure recall; test only, never tune on them |

- Tile score: 99.9th percentile of the anomaly map.
- Metrics: **false alarm rate** (good tiles flagged), **recall** (defect tiles caught). Optional: IoU against the mask.
- Priority: high recall first, then reduce false alarms.

## 8. Results

Fill in per fabric type.

| Fabric | Method | False alarm | Recall | Notes |
|---|---|---|---|---|
| | Background subtraction | | | baseline |
| | FFT notch | | | |
| | Gabor | | | |
| | Top-hat | | | |
| | Combined (best 2-3) | | | pixel-wise max of normalized maps |

## 9. Next steps

1. ~~Fix preprocessing (multi-mask images, mask threshold, stale outputs).~~ Done.
2. ~~Count tiles per fabric code.~~ Done (table in section 4). Pick fabric 02 or 03 for the first test.
3. Check mask overlays on a few defect tiles to confirm tile/mask alignment (mask and image sizes already verified to match).
4. Split good strips of the chosen fabric into `good_cal` / `good_test` by source image.
5. Run the background subtraction baseline and record false alarm and recall.
6. Add FFT, Gabor and the other methods and compare in the table above.
7. Combine the best methods into one anomaly map and produce the colormap figures.
8. Compare the best classical result with a simple ML baseline, then move to Layer 2.

## 10. Requirements

- Python 3, `opencv-python`, `numpy`
- Dataset: AITEX (download separately and extract into `data/raw/`)