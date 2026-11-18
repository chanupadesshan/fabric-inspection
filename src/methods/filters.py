"""All filters and methods for fabric inspection (README section 5).

Only numpy + OpenCV are needed (same as the project .venv). One-class SVM and Isolation Forest
are small numpy implementations (OpenCV 5 has no `ml` module, scikit-learn is not installed).

Convention
----------
Every *method* takes a grayscale tile (uint8 or float) and returns a float32 **anomaly map**
of the same size: higher = more abnormal. Thresholding happens outside (see `threshold_mask`).
Binary methods (adaptive threshold, Otsu, Canny, Hough, LSD) return 0 / 255 maps.
Otsu and adaptive threshold are binarizers (they always mark pixels, even on good fabric), so they
are kept out of METHODS and listed in BINARIZERS; use them on an anomaly map at the decision stage.
Every method has `steps=` = enhancement applied first (default DEFAULT_STEPS).

Sections
--------
1. Enhancement          gaussian, median, flat_field, clahe, enhance
2. Intensity/statistics background_subtraction, local_zscore, adaptive_threshold, otsu
3. Morphology           tophat_blackhat, opening_closing
4. Frequency            fft_notch, gabor_bank, wavelet, periodicity
5. Texture              lbp_anomaly, glcm_anomaly, hog_anomaly, patch_feature_distance
6. Edge / line          sobel, canny, hough_lines, lsd_lines
7. Reference-based      golden_subtraction, ssim_map
8. Classical learning   fit_one_class_svm, fit_isolation_forest, learned_anomaly_map
9. Helpers / registry   tile_score, robust_threshold, threshold_mask, clean_mask,
                        fit_fusion, fuse, display, METHODS, BINARIZERS
"""
import cv2
import numpy as np

DEFAULT_STEPS = ("gaussian",)  # chosen for fabric 02


def _f32(tile):
    return np.asarray(tile, dtype=np.float32)


def _u8(tile):
    return np.clip(tile, 0, 255).astype(np.uint8)


def _odd(k):
    k = int(k)
    return k if k % 2 else k + 1


# =====================================================================
# 1. Enhancement (applied to each tile inside a method)
# =====================================================================
def gaussian(tile, sigma=1.5):
    """Smooth sensor noise and fine yarn texture."""
    return cv2.GaussianBlur(_f32(tile), (0, 0), sigma)


def median(tile, k=3):
    """Remove salt-and-pepper noise while keeping edges (k must be odd)."""
    return cv2.medianBlur(_u8(tile), _odd(k)).astype(np.float32)


def flat_field(tile, sigma=40):
    """Remove uneven lighting: divide by a heavily blurred copy, keep the mean brightness."""
    t = _f32(tile)
    bg = cv2.GaussianBlur(t, (0, 0), sigma)
    return t / (bg + 1e-6) * bg.mean()


def clahe(tile, clip=2.0, grid=8):
    """Boost local contrast so faint defects stand out."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return op.apply(_u8(tile)).astype(np.float32)


STEPS = {"gaussian": gaussian, "median": median, "flat_field": flat_field, "clahe": clahe}


def enhance(tile, steps=("median", "flat_field"), params=None):
    """Apply enhancement steps in order. steps=() = no change.
    params: optional per-step settings, e.g. {"median": {"k": 5}, "clahe": {"clip": 3.0}}"""
    params = params or {}
    t = _f32(tile)
    for s in steps:
        t = np.clip(STEPS[s](t, **params.get(s, {})), 0, 255)
    return t


# =====================================================================
# 2. Intensity / statistics
# =====================================================================
def background_subtraction(tile, steps=DEFAULT_STEPS, bg_sigma=15, exclude=14, return_normal=False):
    """|tile - normal fabric|, normal = heavy blur. Pass 2 repeats the blur without clearly
    abnormal pixels (> exclude), so a defect does not leak into 'normal' and leave a halo."""
    t = enhance(tile, steps)
    normal = cv2.GaussianBlur(t, (0, 0), bg_sigma)
    amap = np.abs(t - normal)
    w = cv2.erode((amap < exclude).astype(np.float32), np.ones((7, 7), np.uint8))
    normal = cv2.GaussianBlur(t * w, (0, 0), bg_sigma) / (cv2.GaussianBlur(w, (0, 0), bg_sigma) + 1e-6)
    amap = np.abs(t - normal).astype(np.float32)
    return (amap, t, normal) if return_normal else amap


def local_zscore(tile, steps=DEFAULT_STEPS, win=31):
    """|pixel - local mean| / local std in a win x win window. > 3 = 3-sigma outlier."""
    t = enhance(tile, steps)
    k = (_odd(win), _odd(win))
    mu = cv2.blur(t, k)
    sd = np.sqrt(np.maximum(cv2.blur(t * t, k) - mu * mu, 1e-6))
    return (np.abs(t - mu) / sd).astype(np.float32)


def adaptive_threshold(tile, steps=DEFAULT_STEPS, block=51, c=10):
    """Binary 0/255: pixels darker OR brighter than their local mean by more than c."""
    t = _u8(enhance(tile, steps))
    b = _odd(block)
    dark = cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b, c)
    bright = cv2.adaptiveThreshold(255 - t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b, c)
    return np.maximum(dark, bright).astype(np.float32)


def otsu_threshold(amap):
    """Otsu threshold of any anomaly map. Returns (bool mask, threshold in map units).
    Note: Otsu always splits the map in two, so on good tiles it still marks pixels."""
    m = _f32(amap)
    lo, hi = float(m.min()), float(m.max())
    u = _u8((m - lo) / (hi - lo + 1e-6) * 255)
    t, _ = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = lo + t / 255 * (hi - lo)
    return m > thr, thr


def otsu(tile, steps=DEFAULT_STEPS, base=None):
    """Binary 0/255: Otsu on a base anomaly map (default: background_subtraction)."""
    base = base or background_subtraction
    mask, _ = otsu_threshold(base(tile, steps))
    return mask.astype(np.float32) * 255


# =====================================================================
# 3. Morphology
# =====================================================================
def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))


def tophat_blackhat(tile, steps=DEFAULT_STEPS, ksize=15):
    """Top-hat (bright spots smaller than ksize) + black-hat (dark spots)."""
    t = enhance(tile, steps)
    k = _ellipse(ksize)
    return (cv2.morphologyEx(t, cv2.MORPH_TOPHAT, k) + cv2.morphologyEx(t, cv2.MORPH_BLACKHAT, k)).astype(np.float32)


def opening_closing(tile, steps=DEFAULT_STEPS, ksize=5, bg_sigma=15):
    """Opening then closing with a weave-sized kernel removes the yarn texture; what remains
    (defects larger than the kernel) is compared with a heavy blur."""
    t = enhance(tile, steps)
    k = _ellipse(ksize)
    smooth = cv2.morphologyEx(cv2.morphologyEx(t, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return np.abs(smooth - cv2.GaussianBlur(smooth, (0, 0), bg_sigma)).astype(np.float32)


# =====================================================================
# 4. Frequency
# =====================================================================
def fft_notch(tile, steps=DEFAULT_STEPS, peak_ratio=4.0, keep_center=8, notch_radius=3, flat_sigma=25,
              pad=64):
    """Remove the periodic weave by notching its peaks in the FFT spectrum; anomaly = what is left
    after also removing slow lighting changes.
    Border handling: the tile is mirror-padded by `pad` px before filtering (no hard edge at the
    tile border) and the weave peaks are found on a Hann-windowed spectrum (no leakage cross)."""
    t = enhance(tile, steps)
    h0, w0 = t.shape
    tp = cv2.copyMakeBorder(t, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = tp.shape
    tp = tp - tp.mean()
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(tp * win)))).astype(np.float32)
    peaks = (mag == cv2.dilate(mag, np.ones((5, 5), np.uint8))) & \
            (mag > cv2.blur(mag, (15, 15)) + np.log(peak_ratio))
    yy, xx = np.ogrid[:h, :w]
    peaks &= ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) > keep_center ** 2
    notch = np.ones((h, w), np.float32)
    for y, x in zip(*np.nonzero(peaks)):
        cv2.circle(notch, (int(x), int(y)), notch_radius, 0, -1)
    notch = cv2.GaussianBlur(notch, (5, 5), 0)
    F = np.fft.fftshift(np.fft.fft2(tp))
    texture_free = np.real(np.fft.ifft2(np.fft.ifftshift(F * notch))).astype(np.float32)
    res = np.abs(texture_free - cv2.GaussianBlur(texture_free, (0, 0), flat_sigma))
    return res[pad:pad + h0, pad:pad + w0].astype(np.float32)


def gabor_kernels(ksize=21, sigma=3.0, lambdas=(6, 10), n_orient=4, gamma=0.5):
    return [cv2.getGaborKernel((ksize, ksize), sigma, np.pi * i / n_orient, lam, gamma, 0, cv2.CV_32F)
            for lam in lambdas for i in range(n_orient)]


def gabor_bank(tile, steps=DEFAULT_STEPS, kernels=None, bg_sigma=15):
    """Local energy of each Gabor filter; anomaly = largest deviation (in std units) from that
    filter's normal energy level. Defects change the local orientation / frequency of the weave."""
    t = enhance(tile, steps)
    kernels = kernels or gabor_kernels()
    out = np.zeros_like(t)
    for k in kernels:
        e = cv2.GaussianBlur(np.abs(cv2.filter2D(t, cv2.CV_32F, k - k.mean())), (0, 0), 3)
        dev = np.abs(e - cv2.GaussianBlur(e, (0, 0), bg_sigma)) / (e.std() + 1e-6)
        out = np.maximum(out, dev)
    return out.astype(np.float32)


def haar_dwt(t):
    """One level of the 2-D Haar wavelet: approximation and horizontal/vertical/diagonal details."""
    t = _f32(t)
    t = t[: t.shape[0] // 2 * 2, : t.shape[1] // 2 * 2]
    a, b, c, d = t[0::2, 0::2], t[0::2, 1::2], t[1::2, 0::2], t[1::2, 1::2]
    return (a + b + c + d) / 4, (a - b + c - d) / 4, (a + b - c - d) / 4, (a - b - c + d) / 4


def wavelet(tile, steps=DEFAULT_STEPS, levels=2, bg_sigma=15):
    """Haar wavelet: the approximation after `levels` levels no longer contains the fine weave;
    anomaly = approximation vs its heavy blur, plus unusual detail energy at the coarsest level."""
    t = enhance(tile, steps)
    h, w = t.shape
    a = t
    for _ in range(levels):
        a, dh, dv, dd = haar_dwt(a)
    approx = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    detail = cv2.resize(np.sqrt(dh ** 2 + dv ** 2 + dd ** 2), (w, h), interpolation=cv2.INTER_LINEAR)
    amap = np.abs(approx - cv2.GaussianBlur(approx, (0, 0), bg_sigma))
    amap += np.abs(detail - cv2.GaussianBlur(detail, (0, 0), bg_sigma))
    return amap.astype(np.float32)


def weave_period(t, min_p=3):
    """Weave repeat (px, py) in pixels from the autocorrelation of the tile."""
    t = _f32(t) - np.mean(t)
    ac = np.real(np.fft.ifft2(np.abs(np.fft.fft2(t)) ** 2))
    h, w = t.shape
    px = int(np.argmax(ac[0, min_p: w // 4]) + min_p)
    py = int(np.argmax(ac[min_p: h // 4, 0]) + min_p)
    return px, py


def periodicity(tile, steps=DEFAULT_STEPS, period=None, smooth=1.5):
    """Normal weave repeats itself: compare each pixel with the pixel one weave period away
    (left/right, up/down). Defects break the repeat. Anomaly = smallest of those differences."""
    t = enhance(tile, steps)
    px, py = period or weave_period(t)
    dx = np.minimum(np.abs(t - np.roll(t, px, 1)), np.abs(t - np.roll(t, -px, 1)))
    dy = np.minimum(np.abs(t - np.roll(t, py, 0)), np.abs(t - np.roll(t, -py, 0)))
    return cv2.GaussianBlur(np.minimum(dx, dy), (0, 0), smooth).astype(np.float32)


# =====================================================================
# 5. Texture (patch features compared with the normal patch)
# =====================================================================
def _patch_coords(shape, patch, stride):
    h, w = shape
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys[-1] != h - patch: ys.append(max(h - patch, 0))
    if xs[-1] != w - patch: xs.append(max(w - patch, 0))
    return [(y, x) for y in ys for x in xs]


def patch_features(t, feature_fn, patch=32, stride=16):
    """Feature vector of every patch. Returns (coords, features [n, d])."""
    coords = _patch_coords(t.shape, patch, stride)
    return coords, np.array([feature_fn(t[y:y + patch, x:x + patch]) for y, x in coords], np.float32)


def scores_to_map(shape, coords, scores, patch):
    """Spread patch scores back to pixels (overlapping patches are averaged)."""
    acc = np.zeros(shape, np.float32)
    cnt = np.zeros(shape, np.float32)
    for (y, x), s in zip(coords, scores):
        acc[y:y + patch, x:x + patch] += s
        cnt[y:y + patch, x:x + patch] += 1
    return acc / np.maximum(cnt, 1)


def lbp_codes(t):
    """Rotation-invariant uniform LBP (8 neighbours, radius 1): codes 0..9."""
    t = _f32(t)
    c = t[1:-1, 1:-1]
    nb = [t[:-2, :-2], t[:-2, 1:-1], t[:-2, 2:], t[1:-1, 2:], t[2:, 2:], t[2:, 1:-1], t[2:, :-2], t[1:-1, :-2]]
    bits = np.stack([(n >= c) for n in nb]).astype(np.int8)
    ones = bits.sum(0)
    transitions = np.abs(np.diff(np.concatenate([bits, bits[:1]]), axis=0)).sum(0)
    return np.where(transitions <= 2, ones, 9)


def lbp_feature(p):
    codes = lbp_codes(p)
    return np.bincount(codes.ravel(), minlength=10)[:10] / codes.size


def glcm(p, levels=16, offsets=((0, 1), (1, 0), (1, 1), (1, -1))):
    """Symmetric, normalized gray-level co-occurrence matrices, one per offset [n_off, L, L]."""
    q = np.clip((_f32(p) * levels / 256).astype(np.int32), 0, levels - 1)
    h, w = q.shape
    mats = []
    for dy, dx in offsets:
        a = q[max(0, -dy): h - max(0, dy), max(0, -dx): w - max(0, dx)]
        b = q[max(0, dy): h - max(0, -dy) if dy < 0 else h, max(0, dx): w - max(0, -dx) if dx < 0 else w]
        b = b[: a.shape[0], : a.shape[1]]
        m = np.bincount((a * levels + b).ravel(), minlength=levels * levels).reshape(levels, levels).astype(np.float64)
        m = m + m.T
        mats.append(m / max(m.sum(), 1))
    return np.array(mats)


def glcm_feature(p, levels=16):
    """Contrast, dissimilarity, homogeneity, energy, correlation, entropy (averaged over offsets)."""
    M = glcm(p, levels)
    i, j = np.mgrid[:levels, :levels]
    feats = []
    for m in M:
        mi, mj = (i * m).sum(), (j * m).sum()
        si, sj = np.sqrt(((i - mi) ** 2 * m).sum()), np.sqrt(((j - mj) ** 2 * m).sum())
        feats.append([((i - j) ** 2 * m).sum(), (np.abs(i - j) * m).sum(), (m / (1 + np.abs(i - j))).sum(),
                      np.sqrt((m ** 2).sum()), ((i - mi) * (j - mj) * m).sum() / (si * sj + 1e-9),
                      -(m[m > 0] * np.log(m[m > 0])).sum()])
    return np.mean(feats, axis=0)


def hog_feature(p, bins=9, cells=2):
    """Histogram of oriented gradients: cells x cells grid, unsigned orientation, L2-normalized."""
    t = _f32(p)
    gx, gy = cv2.Sobel(t, cv2.CV_32F, 1, 0), cv2.Sobel(t, cv2.CV_32F, 0, 1)
    mag, ang = np.sqrt(gx ** 2 + gy ** 2), (np.degrees(np.arctan2(gy, gx)) % 180)
    b = np.minimum((ang / 180 * bins).astype(int), bins - 1)
    h, w = t.shape
    out = []
    for cy in range(cells):
        for cx in range(cells):
            sl = (slice(cy * h // cells, (cy + 1) * h // cells), slice(cx * w // cells, (cx + 1) * w // cells))
            out.append(np.bincount(b[sl].ravel(), mag[sl].ravel(), minlength=bins))
    v = np.concatenate(out)
    return v / (np.linalg.norm(v) + 1e-6)


def stats_feature(p):
    p = _f32(p)
    return np.array([p.mean(), p.std()])


FEATURES = {"stats": stats_feature, "lbp": lbp_feature, "glcm": glcm_feature, "hog": hog_feature}


def combined_feature(names=("stats", "lbp", "glcm", "hog")):
    return lambda p: np.concatenate([FEATURES[n](p) for n in names])


def fit_patch_reference(good_tiles, feature_fn, steps=DEFAULT_STEPS, patch=32, stride=16):
    """Normal patch features from good tiles: median and spread (MAD) of every feature."""
    F = np.vstack([patch_features(enhance(t, steps), feature_fn, patch, stride)[1] for t in good_tiles])
    med = np.median(F, 0)
    return {"median": med, "mad": 1.4826 * np.median(np.abs(F - med), 0) + 1e-6}


def patch_distance_map(tile, feature_fn, steps=DEFAULT_STEPS, patch=32, stride=16, ref=None):
    """Robust distance of each patch's features from normal. Normal = `ref` (from
    fit_patch_reference on good tiles) or, without ref, the median patch of this tile."""
    t = enhance(tile, steps)
    coords, F = patch_features(t, feature_fn, patch, stride)
    if ref is None:
        med = np.median(F, 0)
        ref = {"median": med, "mad": 1.4826 * np.median(np.abs(F - med), 0) + 1e-6}
    z = (F - ref["median"]) / ref["mad"]
    return scores_to_map(t.shape, coords, np.sqrt(np.mean(z ** 2, 1)), patch)


def lbp_anomaly(tile, steps=DEFAULT_STEPS, patch=32, stride=16, ref=None):
    return patch_distance_map(tile, lbp_feature, steps, patch, stride, ref)


def glcm_anomaly(tile, steps=DEFAULT_STEPS, patch=32, stride=16, ref=None):
    return patch_distance_map(tile, glcm_feature, steps, patch, stride, ref)


def hog_anomaly(tile, steps=DEFAULT_STEPS, patch=32, stride=16, ref=None):
    return patch_distance_map(tile, hog_feature, steps, patch, stride, ref)


def patch_feature_distance(tile, steps=DEFAULT_STEPS, patch=32, stride=16, ref=None,
                           features=("stats", "lbp", "glcm", "hog")):
    return patch_distance_map(tile, combined_feature(features), steps, patch, stride, ref)


# =====================================================================
# 6. Edge / line
# =====================================================================
def sobel(tile, steps=DEFAULT_STEPS, smooth=2.0, bg_sigma=15):
    """Gradient magnitude; anomaly = local edge energy that differs from the normal weave's."""
    t = enhance(tile, steps)
    mag = cv2.magnitude(cv2.Sobel(t, cv2.CV_32F, 1, 0), cv2.Sobel(t, cv2.CV_32F, 0, 1))
    e = cv2.GaussianBlur(mag, (0, 0), smooth)
    return np.abs(e - cv2.GaussianBlur(e, (0, 0), bg_sigma)).astype(np.float32)


def canny(tile, steps=DEFAULT_STEPS, low=None, high=None, sigma=0.33):
    """Binary 0/255 edge map. Thresholds default to median-based automatic values.
    The weave itself also gives edges; use stronger enhancement or longer-line filters."""
    t = _u8(enhance(tile, steps))
    v = float(np.median(t))
    low = int(max(0, (1 - sigma) * v)) if low is None else low
    high = int(min(255, (1 + sigma) * v)) if high is None else high
    return cv2.Canny(t, low, high).astype(np.float32)


def _draw_segments(shape, segs, thickness=2):
    out = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return out.astype(np.float32)


def hough_lines(tile, steps=("gaussian", "gaussian"), min_length=60, max_gap=5, votes=40, return_lines=False):
    """Binary 0/255 map of long straight lines (probabilistic Hough on Canny edges): broken yarns,
    thick yarns. Lines shorter than min_length (e.g. the weave) are ignored."""
    edges = _u8(canny(tile, steps))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, votes, minLineLength=min_length, maxLineGap=max_gap)
    segs = [] if lines is None else [tuple(l) for l in np.asarray(lines).reshape(-1, 4)]
    m = _draw_segments(edges.shape, segs)
    return (m, segs) if return_lines else m


def lsd_lines(tile, steps=("gaussian", "gaussian"), min_length=60, return_lines=False):
    """Binary 0/255 map of long line segments from OpenCV's Line Segment Detector
    (falls back to Hough if LSD is not available in this OpenCV build)."""
    if not hasattr(cv2, "createLineSegmentDetector"):
        return hough_lines(tile, steps, min_length, return_lines=return_lines)
    t = _u8(enhance(tile, steps))
    found = cv2.createLineSegmentDetector().detect(t)[0]
    found = [] if found is None else np.asarray(found).reshape(-1, 4)
    segs = [tuple(s) for s in found if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_length]
    m = _draw_segments(t.shape, segs)
    return (m, segs) if return_lines else m


# =====================================================================
# 7. Reference-based (needs a defect-free "golden" tile of the same fabric)
# =====================================================================
def golden_from(good_tiles):
    """Pixel-wise median of good tiles: a simple golden sample (only meaningful if they are aligned)."""
    return np.median(np.stack([_f32(t) for t in good_tiles]), 0).astype(np.float32)


def align(tile, ref, min_response=0.1, max_shift=20):
    """Shift ref onto tile by phase correlation. Skipped when the match is ambiguous
    (periodic weave gives many equally good shifts)."""
    t, r = _f32(tile), _f32(cv2.resize(_f32(ref), (tile.shape[1], tile.shape[0])))
    win = cv2.createHanningWindow(t.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(r, t, win)
    if resp < min_response or max(abs(dx), abs(dy)) > max_shift:
        return r
    return cv2.warpAffine(r, np.float32([[1, 0, dx], [0, 1, dy]]), t.shape[::-1], borderMode=cv2.BORDER_REFLECT)


def golden_subtraction(tile, ref, steps=DEFAULT_STEPS):
    """|tile - aligned, brightness-matched golden sample|."""
    t = enhance(tile, steps)
    r = enhance(align(tile, ref), steps)
    r = (r - r.mean()) / (r.std() + 1e-6) * t.std() + t.mean()
    return np.abs(t - r).astype(np.float32)


def ssim_map(tile, ref, steps=DEFAULT_STEPS, sigma=1.5, return_ssim=False):
    """Structural dissimilarity 1 - SSIM (0 = identical, up to 2) against an aligned golden sample."""
    x = enhance(tile, steps)
    y = enhance(align(tile, ref), steps)
    g = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    d = (1 - s).astype(np.float32)
    return (d, float(s.mean())) if return_ssim else d


# =====================================================================
# 8. Classical learning on texture features (trained on good tiles only)
# =====================================================================
class IsolationForest:
    """Small numpy Isolation Forest. score(X) in (0, 1): higher = more anomalous."""

    def __init__(self, n_trees=100, sample_size=256, seed=0):
        self.n_trees, self.sample_size, self.rng = n_trees, sample_size, np.random.default_rng(seed)

    @staticmethod
    def _c(n):
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 2 else (1.0 if n == 2 else 0.0)

    def _grow(self, X, depth, limit):
        if depth >= limit or len(X) <= 1:
            return ("leaf", len(X))
        f = self.rng.integers(X.shape[1])
        lo, hi = X[:, f].min(), X[:, f].max()
        if lo == hi:
            return ("leaf", len(X))
        s = self.rng.uniform(lo, hi)
        m = X[:, f] < s
        return ("node", f, s, self._grow(X[m], depth + 1, limit), self._grow(X[~m], depth + 1, limit))

    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.psi = min(self.sample_size, len(X))
        limit = int(np.ceil(np.log2(max(self.psi, 2))))
        self.trees = [self._grow(X[self.rng.choice(len(X), self.psi, replace=False)], 0, limit)
                      for _ in range(self.n_trees)]
        return self

    def _path(self, x, node, depth=0):
        while node[0] == "node":
            node = node[3] if x[node[1]] < node[2] else node[4]
            depth += 1
        return depth + self._c(node[1])

    def score(self, X):
        X = np.asarray(X, np.float64)
        h = np.array([[self._path(x, t) for t in self.trees] for x in X]).mean(1)
        return 2 ** (-h / self._c(self.psi))


class OneClassSVM:
    """One-class SVM with an RBF kernel, numpy only (OpenCV 5 has no `ml` module).

    The RBF kernel is approximated with random Fourier features, then a linear one-class SVM
    (Schoelkopf formulation) is trained by subgradient descent on
        1/2 |w|^2 + 1/(nu n) sum max(0, rho - w.z_i) - rho
    About a fraction `nu` of the training patches end up outside the boundary.
    score(X): higher = more anomalous (rho - w.z)."""

    def __init__(self, nu=0.05, gamma=None, n_components=300, n_iter=500, lr=0.5, seed=0):
        self.nu, self.gamma, self.D, self.n_iter, self.lr = nu, gamma, n_components, n_iter, lr
        self.rng = np.random.default_rng(seed)

    def _z(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(np.asarray(X, np.float64) @ self.W + self.b)

    def fit(self, X):
        X = np.asarray(X, np.float64)
        gamma = self.gamma or 1.0 / X.shape[1]
        self.W = self.rng.normal(0, np.sqrt(2 * gamma), (X.shape[1], self.D))
        self.b = self.rng.uniform(0, 2 * np.pi, self.D)
        Z = self._z(X)
        n = len(Z)
        self.w, self.rho = Z.mean(0), 0.0
        for it in range(self.n_iter):
            step = self.lr / np.sqrt(1 + it)
            out = Z @ self.w < self.rho                       # patches outside the boundary
            self.w -= step * (self.w - Z[out].sum(0) / (self.nu * n))
            self.rho -= step * (out.sum() / (self.nu * n) - 1)
        return self

    def score(self, X):
        return (self.rho - self._z(X) @ self.w).astype(np.float32)


def _fit_learned(model, good_tiles, steps, patch, stride, features, max_patches, seed):
    fn = combined_feature(features)
    F = np.vstack([patch_features(enhance(t, steps), fn, patch, stride)[1] for t in good_tiles])
    if len(F) > max_patches:
        F = F[np.random.default_rng(seed).choice(len(F), max_patches, replace=False)]
    mu, sd = F.mean(0), F.std(0) + 1e-6
    model.fit((F - mu) / sd)
    return {"model": model, "mu": mu, "sd": sd, "steps": steps, "patch": patch, "stride": stride,
            "features": features}


def fit_one_class_svm(good_tiles, steps=DEFAULT_STEPS, patch=32, stride=16, nu=0.05, gamma=None,
                      features=("stats", "lbp", "glcm", "hog"), max_patches=5000, seed=0):
    """Train a one-class SVM on patch texture features of good tiles."""
    return _fit_learned(OneClassSVM(nu, gamma), good_tiles, steps, patch, stride, features, max_patches, seed)


def fit_isolation_forest(good_tiles, steps=DEFAULT_STEPS, patch=32, stride=16, n_trees=100,
                         features=("stats", "lbp", "glcm", "hog"), max_patches=5000, seed=0):
    """Train an Isolation Forest on patch texture features of good tiles."""
    return _fit_learned(IsolationForest(n_trees, seed=seed), good_tiles, steps, patch, stride, features,
                        max_patches, seed)


def learned_anomaly_map(tile, fitted):
    """Anomaly map from a model returned by fit_one_class_svm / fit_isolation_forest."""
    t = enhance(tile, fitted["steps"])
    coords, F = patch_features(t, combined_feature(fitted["features"]), fitted["patch"], fitted["stride"])
    s = fitted["model"].score((F - fitted["mu"]) / fitted["sd"])
    return scores_to_map(t.shape, coords, s, fitted["patch"])


# =====================================================================
# 9. Helpers and registry
# =====================================================================
def tile_score(amap, k=25, sigma=2.0, normalize=False, return_parts=False):
    """Tile score for the OK / DAMAGED decision (README section 7): smooth the map (sigma px),
    then take the mean of the k highest values. k ~ smallest real defect, so a 4-20 px defect
    still moves the score (a 99.9th percentile is the ~66th pixel and cannot see it).
    k=1 gives the max after smoothing.

    normalize=True divides by the tile's own typical anomaly level (median of the smoothed map):
    "how many times stronger is the strongest spot than this tile's normal texture response".
    Strips differ in contrast / lighting / weave tightness, which scales the whole map; dividing
    removes that bias and needs only the tile itself. Not for 0 / 255 maps (median is often 0).
    return_parts=True returns (score, peak, level) with peak = top-k mean, level = median."""
    m = _f32(amap)
    if sigma:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    v = m.ravel()
    k = int(min(max(k, 1), v.size))
    peak = float(np.mean(np.partition(v, -k)[-k:]))
    level = max(float(np.median(v)), 1e-6)
    score = peak / level if normalize else peak
    return (score, peak, level) if return_parts else score


def robust_threshold(cal_scores, k=4.0):
    """Threshold from good_cal tile scores only: median + k * 1.4826 * MAD (README section 7).
    Not a multiplicative margin, so it still works when good tiles score ~0 (e.g. top-hat),
    and one odd calibration tile cannot push it up much."""
    c = np.asarray(cal_scores, dtype=np.float64)
    med = float(np.median(c))
    spread = 1.4826 * float(np.median(np.abs(c - med)))
    if spread <= 0:  # all cal scores (almost) equal
        spread = float(c.std()) or 1e-6
    return med + k * spread


def threshold_mask(amap, thr, min_area=4):
    """Damage mask: pixels above thr, connected spots smaller than min_area removed.
    Keep min_area <= 4: the smallest labelled defects are only a few pixels."""
    binary = (_f32(amap) > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    return np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area])


def clean_mask(mask, open_k=3, close_k=3, min_area=4):
    """Opening (remove specks) + closing (fill gaps) + drop spots smaller than min_area."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if open_k: m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(open_k))
    if close_k: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_k))
    return threshold_mask(m, 0.5, min_area)


def _per_tile_level(m):
    """Map divided by its own typical level (median): removes strip-to-strip contrast differences."""
    m = _f32(m)
    return m / max(float(np.median(m)), 1e-6)


def fit_fusion(cal_maps, per_tile=True):
    """Fixed per-method normalisation for fusion, from good_cal maps of ONE fabric.
    cal_maps: {method name: list of anomaly maps on good_cal tiles}.
    per_tile: first divide every map by its own median (same idea as tile_score(normalize=True)).
    Returns {name: (median, p99.9)} of all cal pixels. Use the same per_tile in fuse()."""
    stats = {}
    for name, maps in cal_maps.items():
        v = np.concatenate([(_per_tile_level(m) if per_tile else _f32(m)).ravel() for m in maps])
        stats[name] = (float(np.median(v)), float(np.percentile(v, 99.9)))
    return stats


def fuse(maps, stats, per_tile=True):
    """Combined anomaly map = pixel-wise max of maps normalised with fixed good_cal statistics
    (never per-tile min-max, which would amplify noise on good tiles).
    maps: {name: anomaly map of one tile}; stats: from fit_fusion (same per_tile).
    The result is already normalised per tile, so score it with tile_score(normalize=False)."""
    out = None
    for name, m in maps.items():
        med, hi = stats[name]
        m = _per_tile_level(m) if per_tile else _f32(m)
        z = (m - med) / max(hi - med, 1e-6)
        out = z if out is None else np.maximum(out, z)
    return np.maximum(out, 0).astype(np.float32)


def display(amap, vmax):
    """Fixed-range display image in [0, 1] (README: never per-image min-max).
    Use vmax = a multiple of the cal-derived pixel threshold, so 'hot' means 'would be flagged'."""
    return np.clip(_f32(amap) / vmax, 0, 1)


# Methods that need only the tile (default settings). Reference-based and learned methods
# need extra inputs: golden_subtraction(tile, ref), ssim_map(tile, ref), learned_anomaly_map(tile, fitted).
METHODS = {
    "background_subtraction": background_subtraction,
    "local_zscore": local_zscore,
    "tophat_blackhat": tophat_blackhat,
    "opening_closing": opening_closing,
    "fft_notch": fft_notch,
    "gabor_bank": gabor_bank,
    "wavelet": wavelet,
    "periodicity": periodicity,
    "lbp": lbp_anomaly,
    "glcm": glcm_anomaly,
    "hog": hog_anomaly,
    "patch_features": patch_feature_distance,
    "sobel": sobel,
    "canny": canny,
    "hough": hough_lines,
    "lsd": lsd_lines,
}

# Binarizers: they split any image into two classes, so on good fabric they still mark pixels.
# Not detectors on their own; use them on an anomaly map at the decision stage.
BINARIZERS = {
    "adaptive_threshold": adaptive_threshold,
    "otsu": otsu,
}
