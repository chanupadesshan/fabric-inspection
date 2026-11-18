"""Fabric inspection demo: upload a photo, get DAMAGED / OK, and a report of where each damage is
(left / centre / right, top / middle / bottom) and how big it is (length and width).

Run from the project folder:   streamlit run app/app.py
Uses the same methods as notebook 25: background subtraction, PatchCore and their ensemble.
"""
import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "methods"))
import evaluate as E
import filters as F
import models as M

TRAIN_FABRIC = "02"      # AITEX fabric whose good strips train PatchCore and set the thresholds
FIT_FRACTION = 0.7
LONG_SIDE = 640
WEIGHTS = os.environ.get("FABRIC_WEIGHTS", M.WEIGHTS)   # name or path to a saved state_dict

st.set_page_config(page_title="Fabric inspection", page_icon="🧵", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #1f3a5f 0%, #3b6ea8 60%, #5aa0d6 100%); color: white;
       padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 4px 0 0 0; opacity: .9;}
.verdict {border-radius: 14px; padding: 16px 22px; margin: 6px 0 14px 0; color: white; display: flex;
          align-items: center; gap: 16px;}
.verdict .big {font-size: 1.9rem; font-weight: 800; letter-spacing: .5px;}
.verdict .small {opacity: .92; font-size: .98rem;}
.v-bad {background: linear-gradient(120deg, #b3261e, #e5483f);}
.v-ok {background: linear-gradient(120deg, #1b7f46, #36b36b);}
.v-check {background: linear-gradient(120deg, #b26a00, #e8a025);}
.spot-head {display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem;}
.badge {background: #e5483f; color: white; border-radius: 50%; width: 28px; height: 28px; display: inline-flex;
        align-items: center; justify-content: center; font-weight: 800;}
.chip {display: inline-block; background: rgba(90,160,214,.18); border: 1px solid rgba(90,160,214,.5);
       border-radius: 999px; padding: 2px 12px; margin: 2px 6px 2px 0; font-size: .9rem; font-weight: 600;}
.loc {border-collapse: separate; border-spacing: 3px;}
.loc td {width: 26px; height: 18px; border-radius: 4px; background: rgba(128,128,128,.25);}
.loc td.on {background: #e5483f;}
.kv {display: flex; gap: 22px; margin-top: 8px;}
.kv div {line-height: 1.2;}
.kv .v {font-size: 1.35rem; font-weight: 800;}
.kv .k {font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the models (first start takes a few minutes)...")
def load_models():
    good, _, _, _ = E.collect(TRAIN_FABRIC)
    cal, _ = E.split_by_source(good)
    fit_paths, _ = E.split_by_source(cal, FIT_FRACTION, 1)
    pc = M.fit_patchcore([E.load(p) for p in fit_paths], weights=WEIGHTS)
    members = {
        "background_subtraction": dict(method=lambda t: F.background_subtraction(t)),
        "patchcore": dict(fit=lambda ts: (lambda f: (lambda t: M.patchcore_map(t, f)))(M.fit_patchcore(ts, weights=WEIGHTS))),
    }
    ens = E.calibrate_ensemble(members, TRAIN_FABRIC, "max", fit_fraction=FIT_FRACTION)
    return pc, ens


def starts(n, t=256):
    if n <= t:
        return [0]
    k = int(np.ceil((n - t) / (t * 0.75))) + 1
    return [int(round(i * (n - t) / (k - 1))) for i in range(k)]


def z(m, sigma=2.0):
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    med = np.median(m)
    mad = 1.4826 * np.median(np.abs(m - med)) + 1e-6
    return (m - med) / mad


def heat(gray, amap):
    lo, hi = np.percentile(amap, 50), np.percentile(amap, 99.9)
    u = np.uint8(np.clip((amap - lo) / (hi - lo + 1e-6), 0, 1) * 255)
    return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.45, cv2.applyColorMap(u, cv2.COLORMAP_INFERNO), 0.55, 0)


def load_upload(data, long_side=LONG_SIDE):
    """Returns (bgr, gray, shrink) where shrink = original width / working width."""
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None, None, 1.0
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(256, int(round(h * s / 8)) * 8), max(256, int(round(w * s / 8)) * 8)   # at least one tile
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    return im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), w / nw


def inspect(gray, ens, pc, use_bg, use_pc):
    """Tile decisions + heat map for the chosen methods. Returns (flagged tile corners, n_tiles, heat_map)."""
    maps = []
    if use_bg:
        maps.append(z(F.background_subtraction(gray)))
    if use_pc:
        maps.append(z(M.patchcore_map(gray, pc)))
    amap = np.max(maps, axis=0)
    boxes, n = [], 0
    for y in starts(gray.shape[0]):
        for x in starts(gray.shape[1]):
            _, _, det = E.ensemble_judge(gray[y:y + 256, x:x + 256], ens)
            n += 1
            if (use_bg and det[0]["flagged"]) or (use_pc and det[1]["flagged"]):
                boxes.append((x, y))
    return boxes, n, amap


def where(cx, cy, w, h):
    """Words for a position: (horizontal, vertical, column 0-2, row 0-2)."""
    col = min(2, int(3 * cx / w)); row = min(2, int(3 * cy / h))
    return ("Left", "Centre", "Right")[col], ("Top", "Middle", "Bottom")[row], col, row


def find_spots(amap, boxes, strength, margin, min_area=60, max_spots=6, cm_per_px=(0.0, 0.0), gate=True):
    """Damage spots = the strongest connected areas of the heat map inside flagged tiles.

    strength  share of the photo's strongest response a pixel needs to count (0.2 = sensitive, 0.6 = only the strongest)
    margin    pixels at the photo border that are ignored
    gate      True = only inside the flagged tiles (boxes); False = look in the whole photo
    cm_per_px (x, y) real size of one working pixel; (0, 0) = unknown. Adds length_cm / width_cm to each spot.
    Each spot is a dict: rotated box, length (long side), width (short side), position words, strength.
    """
    H, W = amap.shape
    inside = np.zeros(amap.shape, bool) if gate else np.ones(amap.shape, bool)
    for x, y in boxes:
        inside[y:y + 256, x:x + 256] = True
    a = np.where(inside, amap, 0).astype(np.float32)
    if margin > 0:
        a[:margin] = 0; a[-margin:] = 0; a[:, :margin] = 0; a[:, -margin:] = 0
    peak = float(a.max())
    if peak <= 0:
        return []
    core = (a > max(8.0, strength * peak)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(core, np.ones((15, 15), np.uint8)))
    spots = []
    for i in range(1, n):
        if stats[i][4] < min_area:
            continue
        ys, xs = np.nonzero((lab == i) & (core > 0))
        if len(xs) < 5:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        length, width = max(rw, rh) + 1, min(rw, rh) + 1
        length_cm = width_cm = 0.0
        if cm_per_px[0] > 0 and cm_per_px[1] > 0:       # measure in real units: scale each axis, then fit the box
            (_, _), (cw, ch), _ = cv2.minAreaRect(pts * np.array(cm_per_px, np.float32))
            length_cm, width_cm = max(cw, ch) + max(cm_per_px) / 2, min(cw, ch) + max(cm_per_px) / 2
        hx, vy, col, row = where(cx, cy, W, H)
        bx, by, bw, bh = (int(v) for v in stats[i][:4])
        spots.append(dict(box=cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32), bbox=(bx, by, bw, bh),
                          length=float(length), width=float(width), length_cm=float(length_cm), width_cm=float(width_cm), cx=float(cx), cy=float(cy),
                          h=hx, v=vy, col=col, row=row, peak=float(a[lab == i].max())))
    spots.sort(key=lambda s: -s["peak"])
    spots = spots[:max_spots]
    top = spots[0]["peak"] if spots else 1.0
    for s in spots:
        s["rel"] = s["peak"] / top
    return spots


PROMINENCE_MIN = 2.2   # peak / 99.5th percentile of the map: clean AITEX tiles reach 2.0 at most, isolated blobs 2.7+
PEAK_MIN = 8.0         # and the peak itself must be clearly above background (robust z)


def prominence(amap, margin):
    a = amap[max(margin, 1):-max(margin, 1), max(margin, 1):-max(margin, 1)] if margin > 0 else amap
    peak = float(a.max())
    return peak, peak / max(float(np.percentile(a, 99.5)), 1e-6)


def isolated_spots(amap, margin, cm_per_px):
    """Small, isolated spots that stand far above everything else in the photo, even when no tile is flagged."""
    peak, prom = prominence(amap, margin)
    if peak < PEAK_MIN or prom < PROMINENCE_MIN:
        return []
    out = find_spots(amap, [], 0.5, margin, cm_per_px=cm_per_px, gate=False, max_spots=3)
    for sp in out:
        sp["isolated"] = True
    return out


def merge_spots(spots, extra):
    """Add the extra spots that are not already inside a spot found by the tile check."""
    for e in extra:
        if not any(abs(e["cx"] - s["cx"]) < max(s["bbox"][2], e["bbox"][2]) and abs(e["cy"] - s["cy"]) < max(s["bbox"][3], e["bbox"][3])
                   for s in spots):
            spots.append(e)
    return spots


def four_point(size_in, hole):
    """Four-point system: points for one defect. size_in = its longer side in inches.
    up to 3" = 1 | over 3" to 6" = 2 | over 6" to 9" = 3 | over 9" = 4 | any open / hole = 4."""
    if hole:
        return 4
    if size_in <= 0:
        return 0
    return 1 if size_in <= 3 else 2 if size_in <= 6 else 3 if size_in <= 9 else 4


def locator_html(col, row):
    cells = "".join("<tr>" + "".join(f'<td class="{"on" if (r == row and c == col) else ""}"></td>' for c in range(3)) + "</tr>"
                    for r in range(3))
    return f'<table class="loc">{cells}</table>'


def fmt_size(px, cm, shrink):
    """Size text: real units if a scale was given (cm > 0), else pixels of the original photo."""
    if cm > 0:
        return f"{cm:.1f} cm" if cm >= 1 else f"{cm * 10:.1f} mm"
    return f"{px * shrink:.0f} px"


def crop_zoom(im, s, pad=30, size=220):
    x, y, w, h = s["bbox"]
    x0, y0 = max(0, x - pad), max(0, y - pad); x1, y1 = min(im.shape[1], x + w + pad), min(im.shape[0], y + h + pad)
    c = im[y0:y1, x0:x1].copy()
    cv2.polylines(c, [s["box"] - [x0, y0]], True, (0, 0, 255), 2)
    k = size / max(c.shape[:2])
    return cv2.cvtColor(cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def show(target, arr, **kw):
    """st.image that fills its column on old and new Streamlit versions."""
    try:
        target.image(arr, width="stretch", **kw)
    except TypeError:
        target.image(arr, use_container_width=True, **kw)


def verdict(kind, title, text):
    icon = {"v-bad": "⚠️", "v-ok": "✅", "v-check": "🔎"}[kind]
    st.markdown(f'<div class="verdict {kind}"><div style="font-size:2.2rem">{icon}</div>'
                f'<div><div class="big">{title}</div><div class="small">{text}</div></div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=64)
def inspect_photo(data, use_bg, use_pc, _pc, _ens):
    """Load + inspect one photo. Cached on the photo's bytes and the method, so a rerun (typing a size, moving a
    slider, uploading another photo) does not inspect photos that were already done."""
    im, gray, shrink = load_upload(data)
    if im is None:
        return None
    boxes, n, amap = inspect(gray, _ens, _pc, use_bg, use_pc)
    return im, gray, shrink, boxes, n, amap


# ----------------------------------------------------------------------------- page
st.markdown('<div class="hero"><h1>🧵 Fabric inspection</h1>'
            '<p>Upload a photo of fabric. The app says DAMAGED or OK, and reports where each damage is and how big it is.</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Detector")
    choice = st.radio("Method", ["Ensemble (both)", "Background subtraction", "PatchCore"], index=0,
                      help="Ensemble flags a tile when either method flags it: catches most damage, more false alarms.")
    st.header("Damage spots")
    strength = st.slider("Spot sensitivity", 0.1, 0.9, 0.4, 0.05, help="Lower = more spots are boxed. Higher = only the strongest.")
    margin = st.slider("Ignore photo border (px)", 0, 80, 24, 4, help="Edges of the photo and objects at the border often light up.")
    sensitive = st.checkbox("Also look for small isolated spots", value=True,
                            help="Catches a small lump, hole or stain that stands out clearly in the heat map even when no tile "
                                 "crosses the AITEX threshold. More false alarms possible.")
    st.header("Damage size")
    unit = st.radio("Unit for the size you type", ["cm", "mm"], horizontal=True)
    st.caption("The app finds where the damage is. You measure it (ruler) and type the length and width on each damage card.")
    st.warning("Thresholds come from AITEX fabric 02, not from your fabric. Treat the verdict as a first indication: "
               "chalk, stickers, edges and the table can also be boxed.")

use_bg = choice in ("Ensemble (both)", "Background subtraction")
use_pc = choice in ("Ensemble (both)", "PatchCore")

files = st.file_uploader("Drop fabric photo(s) here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not files:
    st.info("Upload one or more photos to start.")
    st.stop()

pc, ens = load_models()
rows = []   # one row per damage, filled in the loop below

for f in files:
    st.divider()
    with st.spinner(f"Inspecting {f.name} (only the first time)..."):
        result = inspect_photo(f.getvalue(), use_bg, use_pc, pc, ens)
    if result is None:
        st.error(f"{f.name}: could not read this image.")
        continue
    im, gray, shrink, boxes, n, amap = result
    spots = find_spots(amap, boxes, strength, margin) if boxes else []
    if sensitive:
        spots = merge_spots(spots, isolated_spots(amap, margin, (0.0, 0.0)))
    st.subheader(f.name)
    if spots:
        verdict("v-bad", "DAMAGED", f"{len(spots)} suspicious area{'s' if len(spots) != 1 else ''} found "
                                    f"· {len(boxes)} of {n} tiles flagged")
    elif boxes:
        verdict("v-check", "CHECK", f"{len(boxes)} of {n} tiles flagged, but no clear damage spot "
                                    "(only the photo border or weak responses). Look at the heat map.")
    else:
        verdict("v-ok", "OK", f"No tile flagged ({n} tiles checked)")

    marked = im.copy()
    for k, s in enumerate(spots, 1):
        cv2.polylines(marked, [s["box"]], True, (0, 0, 255), 3)
        x, y, _, _ = s["bbox"]
        cv2.circle(marked, (max(x, 14), max(y - 2, 14)), 13, (0, 0, 255), -1)
        cv2.putText(marked, str(k), (max(x, 14) - 6, max(y - 2, 14) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    left, right = st.columns([3, 2], gap="large")
    with left:
        t1, t2, t3 = st.tabs(["Result", "Heat map", "Original"])
        show(t1, cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
        show(t2, cv2.cvtColor(heat(gray, amap), cv2.COLOR_BGR2RGB), caption="Bright = different from normal fabric")
        show(t3, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    with right:
        if not spots:
            st.markdown("#### Damage report")
            st.write("No damage located." if not boxes else "Nothing clear enough to box. Try a lower sensitivity.")
        else:
            st.markdown(f"#### Damage report · {len(spots)} area{'s' if len(spots) != 1 else ''}")
            for k, s in enumerate(spots, 1):
                with st.container(border=True):
                    c_img, c_txt = st.columns([1, 1.6])
                    show(c_img, crop_zoom(im, s))
                    c_txt.markdown(
                        f'<div class="spot-head"><span class="badge">{k}</span>Damage {k}{" · small isolated spot" if s.get("isolated") else ""}</div>'
                        f'<span class="chip">{s["h"]}</span><span class="chip">{s["v"]}</span>'
                        f'<div style="margin-top:6px">{locator_html(s["col"], s["row"])}</div>',
                        unsafe_allow_html=True)
                    i1, i2 = c_txt.columns(2)
                    key = f"{f.name}|{getattr(f, 'size', 0)}|{k}|{unit}"
                    ln = i1.number_input(f"Length ({unit})", min_value=0.0, value=0.0, step=0.1, format="%.1f", key=key + "|len")
                    wd = i2.number_input(f"Width ({unit})", min_value=0.0, value=0.0, step=0.1, format="%.1f", key=key + "|wid")
                    hole = c_txt.checkbox("Open / hole (counts 4 points)", key=key + "|hole")
                    inch = max(ln, wd) / (2.54 if unit == "cm" else 25.4)        # the longer side decides the points
                    pts = four_point(inch, hole)
                    why = ("open / hole" if hole else f"longer side {inch:.2f} in" if inch > 0 else "")
                    nxt = "" if (hole or pts >= 4 or inch <= 0) else (
                        f" · next point above {(3, 6, 9)[pts - 1] * (2.54 if unit == 'cm' else 25.4):.0f} {unit}")
                    c_txt.markdown(f'<span class="chip">{pts} point{"s" if pts != 1 else ""}</span>'
                                   + (f" <small>{why}{nxt}</small>" if why else " <small>type the size to get points</small>"),
                                   unsafe_allow_html=True)
                    area_cm2 = ln * wd * (1.0 if unit == "cm" else 0.01)      # length x width
                    c_txt.markdown(f'<span class="chip">Area {area_cm2:g} cm&sup2;</span>'
                                   f'<span class="chip">{area_cm2 / 10000:.6f} m&sup2;</span>', unsafe_allow_html=True)
                    rows.append({"Photo": f.name, "Damage": k, "Horizontal": s["h"], "Vertical": s["v"],
                                 f"Length ({unit})": ln, f"Width ({unit})": wd, "Area (cm2)": round(area_cm2, 2),
                                 "Area (m2)": round(area_cm2 / 10000, 6), "Longer side (in)": round(inch, 2),
                                 "Open / hole": hole, "Points": pts})
                    st.progress(float(s["rel"]), text=f"Relative strength {s['rel']:.0%}")
            st.caption("Position is where the damage sits in the photo (3 × 3 grid). Measure each damage with a ruler and "
                       "type its length (longest side) and width (shorter side).")


# ----------------------------------------------------------------------------- summary of all damages
if rows:
    import pandas as pd
    st.divider()
    st.subheader("Damage list")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    missing = int((((df[f"Length ({unit})"] == 0) | (df[f"Width ({unit})"] == 0)) & ~df["Open / hole"]).sum())
    if missing:
        st.caption(f"{missing} damage(s) still need a length and width (a hole counts 4 points without a size).")
    st.download_button("Download damage list (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="damage_list.csv", mime="text/csv")

    st.subheader("Four-point score")
    st.markdown("Points per 100 square yards = **total points / roll length** x **36 / fabric width** x **100** "
                "(roll length in yards, fabric width in inches).")
    p1, p2, p3 = st.columns(3)
    roll_len = p1.number_input("Roll length (yards)", min_value=0.0, value=0.0, step=1.0)
    fab_w = p2.number_input("Fabric width (inches)", min_value=0.0, value=0.0, step=1.0)
    total = int(df["Points"].sum())
    p3.metric("Total points", total)
    tot_area = float(df["Area (m2)"].sum())
    st.metric("Total damage area", f"{tot_area:.6f} m\u00b2", help="Sum of length x width of every damage in the list")
    if roll_len > 0 and fab_w > 0:
        score = total / roll_len * 36 / fab_w * 100
        st.metric("Points per 100 sq yd", f"{score:.1f}")
        st.caption(f"{total} / {roll_len:g} x 36 / {fab_w:g} x 100 = {score:.1f}")
        roll_m2 = (roll_len * 0.9144) * (fab_w * 0.0254)
        st.caption(f"Roll area {roll_m2:.2f} m\u00b2; damaged area is {tot_area / roll_m2:.4%} of it.")
    else:
        st.info("Enter the roll length and the fabric width to get the score.")
    st.caption("All photos above are counted as damages of this one roll. Points per defect: up to 3 in = 1, over 3 to 6 in = 2, "
               "over 6 to 9 in = 3, over 9 in = 4, any open or hole = 4. The longer side of the defect decides.")
