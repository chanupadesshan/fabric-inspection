"""Evaluation helpers shared by every experiment notebook (README section 7).

Split good tiles by source strip, threshold from good_cal only, false alarms on good_test and hard
negatives, recall on defect tiles (overall, by size, per strip), AUROC, IoU / Dice, ms per tile,
per-tile CSV + split manifest. Works for any method tile -> anomaly map.
Filters and methods themselves live in filters.py; this file only evaluates them.
"""
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import filters as F

ROOT = Path(__file__).resolve().parents[2]   # project root (src/methods/evaluate.py -> ../../)
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SMALL_DEFECT = 50   # px of mask area: recall is reported separately below / above this


# ---------------------------------------------------------------- files
def fabric_of(p):
    return Path(p).stem.split("_")[2]


def defect_type_of(p):
    return Path(p).stem.split("_")[1]


def source_of(p):
    """Source strip of a tile: '0001_000_00_t05' -> '0001_000_00'."""
    return Path(p).stem.rsplit("_t", 1)[0]


def load(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def load_mask(p, out=OUT):
    mp = out / "mask" / Path(p).name
    return load(mp) > 0 if mp.exists() else None


def blank_fraction(t):
    """Share of saturated pixels: white/black background outside the fabric."""
    return ((t >= 250) | (t <= 5)).mean()


def collect(fabric, out=OUT, max_blank=0.01):
    """Good tiles, hard negatives (clean tiles from defective strips), defect tiles, number skipped.
    Mostly-background tiles are skipped."""
    def tiles(folder):
        return sorted(p for p in (out / folder).glob("*.png") if fabric_of(p) == fabric)
    good_all, hard_all = tiles("good"), tiles("defect_image_clean_tile")
    good = [p for p in good_all if blank_fraction(load(p)) < max_blank]
    hard = [p for p in hard_all if blank_fraction(load(p)) < max_blank]
    skipped = len(good_all) - len(good) + len(hard_all) - len(hard)
    return good, hard, tiles("defect"), skipped


def split_by_source(good, cal_fraction=0.5, seed=0):
    """good_cal / good_test with whole source strips on one side (README: split by source image)."""
    sources = sorted({source_of(p) for p in good})
    order = np.random.default_rng(seed).permutation(len(sources))
    n_cal = max(1, min(len(sources) - 1, int(round(cal_fraction * len(sources)))))
    cal_src = {sources[i] for i in order[:n_cal]}
    return [p for p in good if source_of(p) in cal_src], [p for p in good if source_of(p) not in cal_src]


# -------------------------------------------------------------- metrics
def overlap(pred, mask):
    inter, union = (pred & mask).sum(), (pred | mask).sum()
    total = pred.sum() + mask.sum()
    return (inter / union if union else np.nan), (2 * inter / total if total else np.nan)


def auroc(labels, scores):
    """Area under the ROC curve (rank formula, ties averaged). 1.0 = perfect, 0.5 = random."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        ranks[scores == s] = ranks[scores == s].mean()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def damage_mask(amap, res, pix_thr=None):
    """Pixels whose smoothed anomaly is above the threshold (same smoothing as the tile score).
    pix_thr: threshold for this tile in map units (normalised score: threshold x the tile's level)."""
    sigma = res["settings"]["score_sigma"]
    m = cv2.GaussianBlur(amap, (0, 0), sigma) if sigma else amap
    return F.threshold_mask(m, res["pix_thr"] if pix_thr is None else pix_thr, res["settings"]["min_area"])


def judge(amap, res):
    """Decision for one anomaly map. Returns (score, flagged, pixel threshold of this tile, peak, level).
    With normalize on, score = peak / the tile's own typical level, so the same decision in map units
    is: pixel > threshold x level."""
    st = res["settings"]
    s, peak, level = F.tile_score(amap, st["score_k"], st["score_sigma"], st["normalize"], return_parts=True)
    return s, s > res["thr"], (res["thr"] * level if st["normalize"] else res["thr"]), peak, level


def rate(num, den):
    return num / den if den else np.nan


# ----------------------------------------------------------- evaluation
def _calibrate(method, fabric, cal, fit, thr_k, min_area, score_k, score_sigma, cal_fraction, seed, normalize,
               fit_fraction=None):
    """Fit (if needed) and set the threshold from good_cal tiles only.
    fit_fraction: split the good_cal strips again: this share is used for fitting, the rest for the
    threshold. Needed for models that memorise their training tiles (PatchCore, PaDiM): scored on the
    tiles they were fitted on they look "too normal", and the threshold would come out far too low."""
    fit_tiles = thr_tiles = cal
    if fit is not None and fit_fraction and len({source_of(p) for p in cal}) >= 2:
        fit_tiles, thr_tiles = split_by_source(cal, fit_fraction, seed + 1)
    if fit is not None:
        t0 = time.perf_counter()
        method = fit([load(p) for p in fit_tiles])
        fit_s = time.perf_counter() - t0
    else:
        fit_s = 0.0
    parts = np.array([F.tile_score(method(load(p)), score_k, score_sigma, normalize, return_parts=True)
                      for p in thr_tiles])
    cal_scores = parts[:, 0]
    thr = F.robust_threshold(cal_scores, thr_k)
    settings = dict(score_k=score_k, score_sigma=score_sigma, thr_k=thr_k, min_area=min_area,
                    cal_fraction=cal_fraction, seed=seed, normalize=bool(normalize), fit_fraction=fit_fraction)
    # pix_thr: the threshold in anomaly-map units for a typical good tile (per tile: see judge)
    pix_thr = thr * float(np.median(parts[:, 2])) if normalize else thr
    return dict(fabric=fabric, thr=thr, pix_thr=pix_thr, settings=settings, method=method, fit_s=fit_s,
                cal_scores=cal_scores, n_fit=len(fit_tiles), n_thr=len(thr_tiles),
                fit_sources=sorted({source_of(p) for p in fit_tiles}),
                thr_sources=sorted({source_of(p) for p in thr_tiles}))


def calibrate(method=None, fabric="02", fit=None, thr_k=4.0, min_area=4, score_k=25, score_sigma=2.0,
              cal_fraction=0.5, seed=0, normalize=True, fit_fraction=None, out=OUT):
    """Only the threshold (and fit) for one fabric, no test run. Same split as evaluate()."""
    good, _, _, _ = collect(fabric, out)
    if len({source_of(p) for p in good}) < 2:
        return None
    cal, _ = split_by_source(good, cal_fraction, seed)
    return _calibrate(method, fabric, cal, fit, thr_k, min_area, score_k, score_sigma, cal_fraction, seed,
                      normalize, fit_fraction)


def evaluate(method=None, fabric="02", fit=None, thr_k=4.0, min_area=4, score_k=25, score_sigma=2.0,
             cal_fraction=0.5, seed=0, normalize=True, fit_fraction=None, out=OUT):
    """Run one method on one fabric. Returns None if there are too few tiles.

    method: tile -> anomaly map.
    fit:    for methods that need good fabric first (texture reference, golden sample, learned models,
            fusion): fit(list of good_cal tiles) -> method. Only good_cal tiles are ever used for fitting.
    Tile score = F.tile_score(map, score_k, score_sigma, normalize).
    normalize: divide the score by the tile's own typical anomaly level (removes the bias between
            strips of different contrast). Turn off for 0 / 255 maps (Canny, Hough, LSD).
    Threshold = median + thr_k * 1.4826 * MAD of good_cal tile scores (F.robust_threshold).
    fit_fraction: for models (models/): share of the good_cal strips used for fitting; the threshold
            comes from the remaining good_cal strips (see _calibrate). None = same tiles for both.
    Hard negatives (clean tiles from defective strips) are test-only, like the defect tiles."""
    good, hard, defect, skipped = collect(fabric, out)
    if len({source_of(p) for p in good}) < 2 or not defect:
        return None
    cal, test = split_by_source(good, cal_fraction, seed)
    res = _calibrate(method, fabric, cal, fit, thr_k, min_area, score_k, score_sigma, cal_fraction, seed,
                     normalize, fit_fraction)
    method = res["method"]

    rows = []
    for kind, label, paths in [("good_test", 0, test), ("hard_negative", 0, hard), ("defect", 1, defect)]:
        for p in paths:
            tile = load(p)
            t0 = time.perf_counter()
            amap = method(tile)
            ms = (time.perf_counter() - t0) * 1000
            s, flagged, pix_thr, peak, level = judge(amap, res)
            iou = dice = np.nan
            area = 0
            mask = load_mask(p, out) if label == 1 else None
            if mask is not None:
                area = int(mask.sum())
                iou, dice = overlap(damage_mask(amap, res, pix_thr), mask)
            rows.append(dict(path=p, tile=p.name, source=source_of(p), kind=kind, label=label,
                             type=defect_type_of(p), area=area, peak=peak, level=level, score=s,
                             pred=int(flagged), iou=iou, dice=dice, ms=ms))
    res.update(cal_sources=sorted({source_of(p) for p in cal}),
               test_sources=sorted({source_of(p) for p in test}),
               n_cal=len(cal), n_cal_src=len({source_of(p) for p in cal}),
               n_test_src=len({source_of(p) for p in test}), skipped=skipped, rows=rows)
    return res


def summarize(res):
    r = res["rows"]
    lab = np.array([x["label"] for x in r])
    pred = np.array([x["pred"] for x in r])
    kind = np.array([x["kind"] for x in r])
    area = np.array([x["area"] for x in r])
    tp = int(((pred == 1) & (lab == 1)).sum()); fn = int(((pred == 0) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum()); tn = int(((pred == 0) & (lab == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (tp + fp) and recall else 0.0
    ious = np.array([x["iou"] for x in r if x["label"] == 1], dtype=float)
    det = np.array([x["iou"] for x in r if x["label"] == 1 and x["pred"] == 1], dtype=float)
    mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else np.nan

    gt, hn, d = kind == "good_test", kind == "hard_negative", lab == 1
    small, large = d & (area < SMALL_DEFECT), d & (area >= SMALL_DEFECT)

    # strip level: a strip is flagged if any of its tiles is flagged
    def strips(sel):
        flagged = {}
        for x, s in zip(r, sel):
            if s:
                flagged[x["source"]] = flagged.get(x["source"], False) or bool(x["pred"])
        return flagged
    good_strips, defect_strips = strips(gt), strips(d)

    return dict(fabric=res["fabric"], n_cal=res["n_cal"], n_good_test=int(gt.sum()),
                n_hard=int(hn.sum()), n_defect=tp + fn,
                tp=tp, fn=fn, fp=fp, tn=tn,
                far=rate(int((pred[gt] == 1).sum()), int(gt.sum())),          # good_test only
                far_hard=rate(int((pred[hn] == 1).sum()), int(hn.sum())),     # hard negatives
                far_all=fp / max(fp + tn, 1),
                recall=recall,
                recall_small=rate(int(pred[small].sum()), int(small.sum())), n_small=int(small.sum()),
                recall_large=rate(int(pred[large].sum()), int(large.sum())), n_large=int(large.sum()),
                strip_far=rate(sum(good_strips.values()), len(good_strips)), n_good_strips=len(good_strips),
                strip_recall=rate(sum(defect_strips.values()), len(defect_strips)),
                n_defect_strips=len(defect_strips),
                precision=precision, f1=f1, auroc=auroc(lab, [x["score"] for x in r]),
                ms=float(np.median([x["ms"] for x in r])),
                thr=res["thr"], iou_all=mean(ious), iou_detected=mean(det))


def save_results(res, method_name, out_dir=RESULTS):
    """Per-tile scores CSV + split manifest (which source strips were good_cal / good_test)."""
    d = Path(out_dir) / method_name
    d.mkdir(parents=True, exist_ok=True)
    keys = ["tile", "source", "kind", "label", "type", "area", "peak", "level", "score", "pred", "iou", "dice", "ms"]
    with open(d / f"scores_fabric{res['fabric']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: x[k] for k in keys} for x in res["rows"])
    manifest = dict(method=method_name, fabric=res["fabric"], threshold=res["thr"],
                    cal_scores=res["cal_scores"].tolist(), good_cal=res["cal_sources"],
                    fitted_on=res["fit_sources"], threshold_from=res["thr_sources"],
                    good_test=res["test_sources"], settings=res["settings"])
    (d / f"split_fabric{res['fabric']}.json").write_text(json.dumps(manifest, indent=1))
    return d


# --------------------------------------------------------------- report
def pct(x):
    return "  n/a" if x != x else f"{x:.1%}"


def print_summary(res, S=None):
    S = S or summarize(res)
    print(f"fabric {res['fabric']}: good_cal {S['n_cal']} tiles ({res['n_cal_src']} strips), "
          f"good_test {S['n_good_test']} tiles ({res['n_test_src']} strips), {S['n_hard']} hard negatives, "
          f"{S['n_defect']} defect tiles ({res['skipped']} background tiles skipped)")
    if res["fit_sources"] != res["thr_sources"]:
        print(f"good_cal is split again: fitted on {res['n_fit']} tiles ({len(res['fit_sources'])} strips), "
              f"threshold from {res['n_thr']} tiles ({len(res['thr_sources'])} strips)")
    if res["settings"]["normalize"]:
        print(f"score = strongest spot / the tile's own typical level (normalised);  threshold {res['thr']:.2f}\n")
    else:
        print(f"score = strongest spot (not normalised);  threshold {res['thr']:.2f}\n")
    print("                    predicted OK   predicted DAMAGED")
    print(f"good + hard neg.    {S['tn']:>12}   {S['fp']:>17}")
    print(f"defect              {S['fn']:>12}   {S['tp']:>17}\n")
    print(f"false alarm rate    {pct(S['far'])} good_test, {pct(S['far_hard'])} hard negatives")
    print(f"recall              {pct(S['recall'])}  (small <{SMALL_DEFECT} px: {pct(S['recall_small'])} of {S['n_small']}, "
          f"large: {pct(S['recall_large'])} of {S['n_large']})")
    print(f"strip level         false alarm {pct(S['strip_far'])} of {S['n_good_strips']} good strips, "
          f"recall {pct(S['strip_recall'])} of {S['n_defect_strips']} defect strips")
    print(f"precision           {pct(S['precision'])}")
    print(f"F1                  {S['f1']:.3f}")
    print(f"AUROC               {S['auroc']:.3f}   (threshold-free ranking quality)")
    print(f"mean IoU            {S['iou_all']:.3f} over all defect tiles, {S['iou_detected']:.3f} over detected ones")
    print(f"time                {S['ms']:.1f} ms/tile (median)"
          + (f", fitting on good_cal {res['fit_s']:.1f} s" if res.get("fit_s") else ""))


def print_by_type(res):
    by_type = {}
    for x in res["rows"]:
        if x["label"] == 1:
            by_type.setdefault(x["type"], []).append(x)
    print(f"{'defect':<8}{'tiles':>6}{'found':>7}{'recall':>9}{'mean IoU':>10}{'median area':>13}{'median score':>14}")
    for t in sorted(by_type):
        xs = by_type[t]
        found = sum(x["pred"] for x in xs)
        ious = np.array([x["iou"] for x in xs], dtype=float)
        iou = np.nanmean(ious) if np.isfinite(ious).any() else np.nan
        print(f"{t:<8}{len(xs):>6}{found:>7}{found / len(xs):>9.0%}{iou:>10.2f}"
              f"{np.median([x['area'] for x in xs]):>13.0f}{np.median([x['score'] for x in xs]):>14.2f}")


def print_fabric_table(summaries):
    hdr = (f"{'fabric':<7}{'test':>6}{'hard':>6}{'defect':>7}{'FAR':>7}{'FAR hn':>8}{'recall':>8}"
           f"{'small':>7}{'large':>7}{'stripFAR':>9}{'stripRec':>9}{'AUROC':>7}{'IoU':>6}{'ms':>6}")
    print(hdr); print("-" * len(hdr))
    for s in summaries:
        print(f"{s['fabric']:<7}{s['n_good_test']:>6}{s['n_hard']:>6}{s['n_defect']:>7}{pct(s['far']):>7}"
              f"{pct(s['far_hard']):>8}{pct(s['recall']):>8}{pct(s['recall_small']):>7}{pct(s['recall_large']):>7}"
              f"{pct(s['strip_far']):>9}{pct(s['strip_recall']):>9}{s['auroc']:>7.3f}{s['iou_all']:>6.2f}{s['ms']:>6.1f}")
    if summaries:
        t = {k: sum(s[k] for s in summaries) for k in ["tp", "fn", "fp", "tn"]}
        print("-" * len(hdr))
        print(f"{'all':<7}{'':>6}{'':>6}{t['tp'] + t['fn']:>7}{'':>7}{'':>8}"
              f"{pct(t['tp'] / max(t['tp'] + t['fn'], 1)):>8}   (FAR all negatives: "
              f"{pct(t['fp'] / max(t['fp'] + t['tn'], 1))})")


def plot_scores_roc(res, S=None):
    """Left: tile-score histograms + threshold. Right: ROC curve (on test, for analysis only)."""
    S = S or summarize(res)
    rows = res["rows"]
    g = np.array([x["score"] for x in rows if x["kind"] == "good_test"])
    h = np.array([x["score"] for x in rows if x["kind"] == "hard_negative"])
    d = np.array([x["score"] for x in rows if x["label"] == 1])
    neg = np.r_[g, h]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    hi = np.percentile(np.r_[neg, d, res["cal_scores"]], 99.5)  # a few huge scores must not squash the plot
    lo = min(0.0, float(np.r_[neg, d, res["cal_scores"]].min()))
    bins = np.linspace(lo, lo + (max(hi, res["thr"]) - lo) * 1.1, 40)
    clip = lambda a: np.minimum(a, bins[-1])
    g, h, d_plot, cal_plot = clip(g), clip(h), clip(d), clip(res["cal_scores"])
    ax[0].hist(cal_plot, bins, alpha=0.4, label="good_cal", color="gray")
    ax[0].hist(g, bins, alpha=0.6, label="good_test", color="tab:blue")
    if len(h):
        ax[0].hist(h, bins, alpha=0.5, label="hard negatives", color="tab:green")
    ax[0].hist(d_plot, bins, alpha=0.6, label="defect", color="tab:red")
    ax[0].axvline(res["thr"], color="k", ls="--", label=f"threshold {res['thr']:.2f}")
    ax[0].set_xlabel(f"tile score (mean of top {res['settings']['score_k']} after smoothing"
                     + (", / the tile's own level)" if res["settings"]["normalize"] else ")"))
    ax[0].set_ylabel("tiles")
    ax[0].legend()
    ts = np.unique(np.r_[neg, d])[::-1]
    ax[1].plot([0] + [(neg >= t).mean() for t in ts], [0] + [(d >= t).mean() for t in ts], lw=2)
    ax[1].plot([0, 1], [0, 1], color="gray", ls=":")
    ax[1].scatter([S["far_all"]], [S["recall"]], color="k", zorder=3, label="chosen threshold (from good_cal)")
    ax[1].set_xlabel("false alarm rate (good_test + hard negatives)"); ax[1].set_ylabel("recall")
    ax[1].set_title(f"ROC  (AUROC {S['auroc']:.3f})"); ax[1].legend(loc="lower right")
    plt.tight_layout(); plt.show()


def failures(res):
    """Missed defects (lowest score first) and false alarms (highest score first)."""
    rows = res["rows"]
    missed = sorted([x for x in rows if x["label"] == 1 and not x["pred"]], key=lambda x: x["score"])
    false_alarms = sorted([x for x in rows if x["label"] == 0 and x["pred"]], key=lambda x: -x["score"])
    return missed, false_alarms


def example_tiles(fabric, out=OUT):
    """Largest defect, smallest defect (by mask area) and last good tile."""
    good, _, defect, _ = collect(fabric, out)
    areas = {p: int(load_mask(p, out).sum()) for p in defect if load_mask(p, out) is not None}
    big, small = max(areas, key=areas.get), min(areas, key=areas.get)
    return [("largest defect", big, areas[big]), ("smallest defect", small, areas[small]),
            ("good tile", good[-1], 0)]


def verdict(is_defect, flagged):
    """Plain-language result for one tile, and its colour."""
    if is_defect:
        return ("DAMAGE CAUGHT", "green") if flagged else ("DAMAGE MISSED", "red")
    return ("FALSE ALARM (good fabric flagged)", "red") if flagged else ("CORRECT: good fabric, OK", "green")


def show_detection(p, res, title=None):
    """3 panels: tile, anomaly map (fixed range: mid-scale = threshold), result with mask outline.
    The header says in words whether the damage was caught (defect tiles) or whether good fabric
    was flagged by mistake (good tiles)."""
    tile = load(p)
    amap = res["method"](tile)
    s, flagged, pix_thr, _, _ = judge(amap, res)
    keep = damage_mask(amap, res, pix_thr)
    mask = load_mask(p)  # only for checking
    is_defect = mask is not None
    text, colour = verdict(is_defect, flagged)

    fig, ax = plt.subplots(1, 3, figsize=(11, 4.3))
    ax[0].imshow(tile, cmap="gray", vmin=0, vmax=255); ax[0].set_title("1. tile")
    ax[1].imshow(F.display(amap, 2 * pix_thr), cmap="inferno", vmin=0, vmax=1)
    ax[1].set_title("2. anomaly map")
    overlay = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    overlay[keep] = [255, 0, 0]
    ax[2].imshow(overlay)
    panel = f"3. method says {'DAMAGED' if flagged else 'OK'}\n(score {s:.2f} vs threshold {res['thr']:.2f}"
    panel += ", normalised)" if res["settings"]["normalize"] else ")"
    if is_defect:
        ax[2].contour(mask, levels=[0.5], colors="lime", linewidths=0.7)
        panel += f"\nred = detected, green = real damage, IoU {overlap(keep, mask)[0]:.2f}"
    else:
        panel += "\nred = detected (no real damage here)"
    ax[2].set_title(panel, fontsize=8.5)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{text}\n{title or Path(p).name}", color=colour, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


def show_failures(res, n=3):
    """Lowest-scoring missed defects and highest-scoring false alarms."""
    missed, false_alarms = failures(res)
    print(f"{len(missed)} missed defects, {len(false_alarms)} false alarms")
    for x in missed[:n] + false_alarms[:n]:
        show_detection(x["path"], res)


def show_examples(res, out=OUT):
    for name, p, area in example_tiles(res["fabric"], out):
        print(f"{name}: {p.name}" + (f", {area} px" if area else ""))
        show_detection(p, res)


def examples_per_type(out=OUT):
    """One defect tile per defect type (second part of the file name): the tile with the largest mask
    area (clearest example), taken from the fabric that has the most tiles of that type."""
    by_type = {}
    for p in sorted((out / "defect").glob("*.png")):
        by_type.setdefault(defect_type_of(p), []).append(p)
    picks = []
    for t, paths in sorted(by_type.items()):
        counts = {}
        for p in paths:
            counts[fabric_of(p)] = counts.get(fabric_of(p), 0) + 1
        fab = max(sorted(counts), key=counts.get)
        cand = [p for p in paths if fabric_of(p) == fab]
        areas = {p: int(load_mask(p, out).sum()) for p in cand}
        best = max(areas, key=areas.get)
        picks.append(dict(type=t, fabric=fab, path=best, area=areas[best], n_tiles=len(paths),
                          fabrics=sorted(counts)))
    return picks


def show_per_type(method=None, fit=None, res=None, show=True, out=OUT, **kw):
    """One example of every defect type, each judged with the threshold of its own fabric
    (calibrated from that fabric's good_cal only; reuses `res` for its fabric)."""
    picks = examples_per_type(out)
    cache = {res["fabric"]: res} if res is not None else {}
    print(f"{'type':<6}{'fabric':>7}{'tiles':>7}  {'example tile':<22}{'area px':>8}{'score':>9}{'thr':>9}  result")
    shown = []
    for x in picks:
        f = x["fabric"]
        if f not in cache:
            cache[f] = calibrate(method, f, fit, out=out, **kw)
        r = cache[f]
        if r is None:
            print(f"{x['type']:<6}{f:>7}  fabric {f}: not enough good strips")
            continue
        s, found = judge(r["method"](load(x["path"])), r)[:2]
        print(f"{x['type']:<6}{f:>7}{x['n_tiles']:>7}  {x['path'].name:<22}{x['area']:>8}{s:>9.2f}{r['thr']:>9.2f}  "
              f"{'DAMAGED (caught)' if found else 'OK (missed)'}")
        shown.append((x, r, found))
    print(f"\ncaught {sum(found for _, _, found in shown)} of {len(shown)} defect types")
    if show:
        for x, r, _ in shown:
            show_detection(x["path"], r, title=f"defect type {x['type']} (fabric {x['fabric']}): {x['path'].name}")


def all_fabrics(method=None, fit=None, method_name=None, out=OUT, **kw):
    """Same method and settings on every fabric (one threshold per fabric). Saves results if named."""
    fabrics = sorted({fabric_of(p) for d in ["good", "defect"] for p in (out / d).glob("*.png")})
    summaries = []
    for f in fabrics:
        r = evaluate(method, f, fit=fit, out=out, **kw)
        if r is None:
            print(f"fabric {f}: skipped (needs >= 2 good strips and >= 1 defect tile)")
            continue
        summaries.append(summarize(r))
        if method_name:
            save_results(r, method_name)
    print_fabric_table(summaries)
    return summaries


def tune(make_fit, base, grid, fabric, **kw):
    """Hyperparameter comparison for a model on ONE fabric (the development fabric).

    make_fit(**params) -> fit function; base = current params; grid = list of dicts, each a change to base.
    Every row is a full evaluate() run. Choosing a setting from this table uses the defect tiles of this
    fabric, so its numbers become optimistic: judge the chosen setting on the OTHER fabrics."""
    print(f"fabric {fabric} (development fabric)")
    hdr = f"{'change to PARAMS':<44}{'FAR':>7}{'FAR hn':>8}{'recall':>8}{'small':>7}{'large':>7}{'AUROC':>7}{'IoU':>6}{'ms':>7}{'fit s':>7}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for change in grid:
        res = evaluate(None, fabric, fit=make_fit(**{**base, **change}), **kw)
        S = summarize(res)
        label = ", ".join(f"{k}={v}" for k, v in change.items()) or "(none: current PARAMS)"
        print(f"{label[:43]:<44}{pct(S['far']):>7}{pct(S['far_hard']):>8}{pct(S['recall']):>8}{pct(S['recall_small']):>7}"
              f"{pct(S['recall_large']):>7}{S['auroc']:>7.3f}{S['iou_all']:>6.2f}{S['ms']:>7.0f}{res['fit_s']:>7.0f}")
        rows.append(dict(change=change, label=label, **S))
    best = max(rows, key=lambda r: r["auroc"])
    print(f"\nhighest AUROC: {best['label']}")
    return rows


# ------------------------------------------------------------- ensemble
# Several members judge the same tile; one rule combines them. Every member's tile score is turned into
# z = (score - median of its good_cal scores) / (1.4826 * MAD), "robust deviations above good fabric", so a
# member flags a tile when z > thr_k (the same decision as its own threshold in evaluate()). The rule then
# combines the z values: max = damaged if ANY member flags, mean = average, min = damaged only if ALL flag.
RULES = {"max": np.max, "mean": np.mean, "min": np.min}


def _combine(rule, z):
    """z: sequence of member z values (or array members x tiles) -> combined score."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {sorted(RULES)}, got {rule!r}")
    return RULES[rule](np.asarray(z, dtype=float), axis=0)


def _rule_thr(rule, res):
    """Threshold of a rule on the combined z scale. max: thr_k (= every member's own threshold).
    mean / min: median + thr_k * 1.4826 * MAD of the combined z on good_cal (same recipe as evaluate())."""
    thr_k = res["settings"]["thr_k"]
    if rule == "max":
        return float(thr_k)
    return F.robust_threshold(_combine(rule, [res["cal_z"][n] for n in res["members"]]), thr_k)


def _calibrate_ensemble(members, fabric, cal, rule, thr_k, min_area, score_k, score_sigma, cal_fraction, seed,
                        fit_fraction):
    """Fit and calibrate every member on good_cal only, then the combined threshold."""
    _combine(rule, [0.0])  # validates the rule
    ress, fit_s = {}, 0.0
    for name, m in members.items():
        r = _calibrate(m.get("method"), fabric, cal, m.get("fit"), thr_k, min_area, score_k, score_sigma,
                       cal_fraction, seed, True, fit_fraction)
        med = float(np.median(r["cal_scores"]))
        r["med"], r["spread"] = med, ((r["thr"] - med) / thr_k if thr_k else 1.0)
        ress[name] = r
        fit_s += r["fit_s"]
    # tiles that no member was fitted on: the combined threshold is set from these
    common = set.intersection(*[set(r["thr_sources"]) for r in ress.values()])
    thr_tiles = [p for p in cal if source_of(p) in common]
    cal_z = {n: [] for n in ress}
    for p in thr_tiles:
        tile = load(p)
        for n, r in ress.items():
            s = F.tile_score(r["method"](tile), score_k, score_sigma, True)
            cal_z[n].append((s - r["med"]) / r["spread"])
    cal_z = {n: np.array(v) for n, v in cal_z.items()}
    settings = dict(score_k=score_k, score_sigma=score_sigma, thr_k=thr_k, min_area=min_area,
                    cal_fraction=cal_fraction, seed=seed, normalize=True, fit_fraction=fit_fraction, rule=rule,
                    members=list(members))
    fitted = [r for r in ress.values() if r["fit_sources"] != r["thr_sources"]] or list(ress.values())
    res = dict(fabric=fabric, members=ress, rule=rule, settings=settings, cal_z=cal_z, fit_s=fit_s, method=None,
               n_fit=fitted[0]["n_fit"], n_thr=len(thr_tiles), fit_sources=fitted[0]["fit_sources"],
               thr_sources=sorted(common))
    res["thr"] = _rule_thr(rule, res)
    res["pix_thr"] = None
    res["cal_scores"] = _combine(rule, [cal_z[n] for n in ress])
    return res


def calibrate_ensemble(members, fabric="02", rule="max", thr_k=4.0, min_area=4, score_k=25, score_sigma=2.0,
                       cal_fraction=0.5, seed=0, fit_fraction=None, out=OUT):
    """Only fit + thresholds of the ensemble for one fabric, no test run. Same split as evaluate_ensemble()."""
    good, _, _, _ = collect(fabric, out)
    if len({source_of(p) for p in good}) < 2:
        return None
    cal, _ = split_by_source(good, cal_fraction, seed)
    return _calibrate_ensemble(members, fabric, cal, rule, thr_k, min_area, score_k, score_sigma, cal_fraction,
                               seed, fit_fraction)


def _judge_ensemble(tile, res):
    """Run every member on one tile. Returns dict: z, flags (per member), score, pred, keep (damage mask:
    union of the flagging members' masks), maps, pix_thr, ms."""
    z, flags, maps, pix, ms = {}, {}, {}, {}, 0.0
    for n, r in res["members"].items():
        t0 = time.perf_counter()
        amap = r["method"](tile)
        ms += (time.perf_counter() - t0) * 1000
        s, fl, pix[n], _, _ = judge(amap, r)
        z[n], flags[n], maps[n] = (s - r["med"]) / r["spread"], bool(fl), amap
    score = float(_combine(res["rule"], list(z.values())))
    pred = score > res["thr"]
    chosen = [n for n in z if flags[n]] or ([max(z, key=z.get)] if pred else [])
    keep = np.zeros(tile.shape, bool)
    for n in chosen:
        keep |= damage_mask(maps[n], res["members"][n], pix[n])
    return dict(z=z, flags=flags, score=score, pred=pred, keep=keep, maps=maps, pix_thr=pix, ms=ms)


def ensemble_judge(tile, res):
    """Decision of a calibrated ensemble (calibrate_ensemble / evaluate_ensemble) for one tile.
    Returns (combined score, flagged, details): details = one dict per member, in member order, with
    name, z, flagged."""
    j = _judge_ensemble(tile, res)
    det = [dict(name=n, z=j["z"][n], flagged=j["flags"][n]) for n in res["members"]]
    return j["score"], bool(j["pred"]), det


def evaluate_ensemble(members, fabric="02", rule="max", thr_k=4.0, min_area=4, score_k=25, score_sigma=2.0,
                      cal_fraction=0.5, seed=0, fit_fraction=None, out=OUT):
    """evaluate() for an ensemble. members: {name: dict(method=tile -> map)} for classical methods or
    {name: dict(fit=good tiles -> method)} for models. Same split, same test tiles as evaluate().
    Rows have the same fields as evaluate() plus `z` (member -> z) and `flags` (member -> flagged).
    Returns None if there are too few tiles."""
    good, hard, defect, skipped = collect(fabric, out)
    if len({source_of(p) for p in good}) < 2 or not defect:
        return None
    cal, test = split_by_source(good, cal_fraction, seed)
    res = _calibrate_ensemble(members, fabric, cal, rule, thr_k, min_area, score_k, score_sigma, cal_fraction,
                              seed, fit_fraction)
    rows = []
    for kind, label, paths in [("good_test", 0, test), ("hard_negative", 0, hard), ("defect", 1, defect)]:
        for p in paths:
            j = _judge_ensemble(load(p), res)
            iou = dice = np.nan
            area = 0
            mask = load_mask(p, out) if label == 1 else None
            if mask is not None:
                area = int(mask.sum())
                iou, dice = overlap(j["keep"], mask)
            rows.append(dict(path=p, tile=p.name, source=source_of(p), kind=kind, label=label,
                             type=defect_type_of(p), area=area, peak=np.nan, level=np.nan, score=j["score"],
                             pred=int(j["pred"]), iou=iou, dice=dice, ms=j["ms"], z=j["z"], flags=j["flags"]))
    res.update(cal_sources=sorted({source_of(p) for p in cal}), test_sources=sorted({source_of(p) for p in test}),
               n_cal=len(cal), n_cal_src=len({source_of(p) for p in cal}),
               n_test_src=len({source_of(p) for p in test}), skipped=skipped, rows=rows)
    return res


def _variants(res):
    """(name, summary) for every member alone (z > thr_k) and every rule, on the same test tiles."""
    thr_k, out = res["settings"]["thr_k"], []
    configs = [(f"{n} alone", (lambda x, n=n: x["z"][n]), thr_k) for n in res["members"]]
    configs += [(f"rule {r}", (lambda x, r=r: float(_combine(r, list(x["z"].values())))), _rule_thr(r, res))
                for r in RULES]
    for name, score, thr in configs:
        rows = [dict(x, score=score(x), pred=int(score(x) > thr)) for x in res["rows"]]
        out.append((name, summarize(dict(res, rows=rows, thr=thr))))
    return out


def ensemble_table(res):
    """Every member alone and every rule on the same test tiles. * = the rule used (RULE)."""
    hdr = f"{'':<2}{'configuration':<30}{'FAR':>7}{'FAR hn':>8}{'recall':>8}{'small':>7}{'large':>7}{'AUROC':>7}"
    print(hdr); print("-" * len(hdr))
    table = _variants(res)
    for name, S in table:
        mark = "*" if name == f"rule {res['rule']}" else ""
        print(f"{mark:<2}{name:<30}{pct(S['far']):>7}{pct(S['far_hard']):>8}{pct(S['recall']):>8}"
              f"{pct(S['recall_small']):>7}{pct(S['recall_large']):>7}{S['auroc']:>7.3f}")
    print(f"\nFAR = false alarm rate on good_test, FAR hn = on clean tiles of defective strips. "
          f"fabric {res['fabric']}, {len(res['members'])} members, rule = {res['rule']}")
    return table


def save_ensemble(res, method_name, out_dir=RESULTS):
    """Per-tile scores CSV (with every member's z) + split manifest."""
    d = Path(out_dir) / method_name
    d.mkdir(parents=True, exist_ok=True)
    names = list(res["members"])
    keys = ["tile", "source", "kind", "label", "type", "area", "score", "pred", "iou", "dice", "ms"]
    with open(d / f"scores_fabric{res['fabric']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + [f"z_{n}" for n in names])
        w.writeheader()
        for x in res["rows"]:
            w.writerow({**{k: x[k] for k in keys}, **{f"z_{n}": x["z"][n] for n in names}})
    manifest = dict(method=method_name, fabric=res["fabric"], rule=res["rule"], threshold=res["thr"],
                    member_thresholds={n: dict(score_thr=r["thr"], median=r["med"], spread=r["spread"])
                                       for n, r in res["members"].items()},
                    cal_scores=res["cal_scores"].tolist(), good_cal=res["cal_sources"],
                    fitted_on=res["fit_sources"], threshold_from=res["thr_sources"],
                    good_test=res["test_sources"], settings=res["settings"])
    (d / f"split_fabric{res['fabric']}.json").write_text(json.dumps(manifest, indent=1))
    return d


def show_ensemble_detection(p, res, title=None):
    """Tile, every member's anomaly map (fixed range, mid-scale = its threshold) with its flag, and the result."""
    tile = load(p)
    j = _judge_ensemble(tile, res)
    mask = load_mask(p)
    is_defect = mask is not None
    text, colour = verdict(is_defect, j["pred"])
    names = list(res["members"])
    fig, ax = plt.subplots(1, len(names) + 2, figsize=(3.6 * (len(names) + 2), 4.3))
    ax[0].imshow(tile, cmap="gray", vmin=0, vmax=255); ax[0].set_title("tile")
    for a, n in zip(ax[1:], names):
        a.imshow(F.display(j["maps"][n], 2 * j["pix_thr"][n]), cmap="inferno", vmin=0, vmax=1)
        a.set_title(f"{n}\n{'FLAGS' if j['flags'][n] else 'ok'}  (z {j['z'][n]:.1f})", fontsize=9,
                    color="red" if j["flags"][n] else "black")
    overlay = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    overlay[j["keep"]] = [255, 0, 0]
    ax[-1].imshow(overlay)
    panel = f"ensemble ({res['rule']}) says {'DAMAGED' if j['pred'] else 'OK'}\nscore {j['score']:.1f} vs thr {res['thr']:.1f}"
    if is_defect:
        ax[-1].contour(mask, levels=[0.5], colors="lime", linewidths=0.7)
        panel += f"\nred = detected, green = real damage, IoU {overlap(j['keep'], mask)[0]:.2f}"
    ax[-1].set_title(panel, fontsize=8.5)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{text}\n{title or Path(p).name}", color=colour, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


def show_ensemble_failures(res, n=3):
    """Lowest-scoring missed defects and highest-scoring false alarms of the ensemble."""
    missed, false_alarms = failures(res)
    print(f"{len(missed)} missed defects, {len(false_alarms)} false alarms")
    for x in missed[:n] + false_alarms[:n]:
        show_ensemble_detection(x["path"], res)


def ensemble_per_type(members, res=None, rule="max", show=True, out=OUT, **kw):
    """One example of every defect type, judged by an ensemble calibrated on its own fabric
    (reuses `res` for its fabric). The table says which members flagged it."""
    kw = {k: v for k, v in kw.items() if k in ("thr_k", "min_area", "score_k", "score_sigma", "cal_fraction",
                                                "seed", "fit_fraction")}
    cache = {res["fabric"]: res} if res is not None else {}
    print(f"{'type':<6}{'fabric':>7}{'tiles':>7}  {'example tile':<22}{'area px':>8}{'score':>8}{'thr':>7}  result")
    shown = []
    for x in examples_per_type(out):
        f = x["fabric"]
        if f not in cache:
            cache[f] = calibrate_ensemble(members, f, rule, out=out, **kw)
        r = cache[f]
        if r is None:
            print(f"{x['type']:<6}{f:>7}  fabric {f}: not enough good strips")
            continue
        j = _judge_ensemble(load(x["path"]), r)
        who = ", ".join(n for n in j["flags"] if j["flags"][n]) or "none"
        print(f"{x['type']:<6}{f:>7}{x['n_tiles']:>7}  {x['path'].name:<22}{x['area']:>8}{j['score']:>8.1f}"
              f"{r['thr']:>7.1f}  {'DAMAGED (caught)' if j['pred'] else 'OK (missed)':<17} members flagging: {who}")
        shown.append((x, r, j["pred"]))
    print(f"\ncaught {sum(found for _, _, found in shown)} of {len(shown)} defect types")
    if show:
        for x, r, _ in shown:
            show_ensemble_detection(x["path"], r, title=f"defect type {x['type']} (fabric {x['fabric']}): {x['path'].name}")


def ensemble_all_fabrics(members, method_name=None, rule="max", out=OUT, **kw):
    """Same ensemble on every fabric (fitted and calibrated per fabric). Prints the ensemble (chosen rule) per
    fabric, then every member alone and every rule pooled over all fabrics. Saves results if named."""
    fabrics = sorted({fabric_of(p) for d in ["good", "defect"] for p in (out / d).glob("*.png")})
    summaries, pooled = [], {}
    for f in fabrics:
        r = evaluate_ensemble(members, f, rule, out=out, **kw)
        if r is None:
            print(f"fabric {f}: skipped (needs >= 2 good strips and >= 1 defect tile)")
            continue
        variants = dict(_variants(r))
        summaries.append(variants[f"rule {rule}"])
        for name, S in variants.items():
            pooled.setdefault(name, []).append(S)
        if method_name:
            save_ensemble(r, method_name)
    print_fabric_table(summaries)
    print(f"\nall fabrics together (rule used: {rule})")
    hdr = f"{'':<2}{'configuration':<30}{'recall':>8}{'FAR all':>9}{'precision':>10}{'F1':>7}{'mean AUROC':>12}"
    print(hdr); print("-" * len(hdr))
    for name, Ss in pooled.items():
        t = {k: sum(s[k] for s in Ss) for k in ["tp", "fn", "fp", "tn"]}
        rec = t["tp"] / max(t["tp"] + t["fn"], 1)
        prec = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else np.nan
        f1 = 2 * prec * rec / (prec + rec) if t["tp"] and prec == prec else 0.0
        mark = "*" if name == f"rule {rule}" else ""
        print(f"{mark:<2}{name:<30}{pct(rec):>8}{pct(t['fp'] / max(t['fp'] + t['tn'], 1)):>9}{pct(prec):>10}"
              f"{f1:>7.3f}{np.nanmean([s['auroc'] for s in Ss]):>12.3f}")
    return summaries
