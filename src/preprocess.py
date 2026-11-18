# Split fabric strips into labelled tiles and create a CSV index.
# Running this script replaces previously generated output tiles.
import csv
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Find data relative to this file, regardless of the current working folder.
ROOT = Path(__file__).resolve().parents[1]   # project root (src/preprocess.py -> ../)
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
# Tile width in pixels; the full image height is preserved.
TILE = 256
MASK_THRESH = 127  # masks have soft (anti-aliased) edges; ignore faint edge pixels
CLEAN_MARGIN = 8   # a tile from a defective image is "clean" only if the mask grown by this many px
                   # does not touch it (soft mask fringes below MASK_THRESH can spill over)
SUBDIRS = ["good", "defect", "defect_image_clean_tile", "mask", "edge"]

# ---- edge / background detection (per 8-px column band of the whole strip) ----
BAND = 8
SAT_FRAC = 0.01   # band with >= 1 % pure white/black pixels = background
STD_FACTOR = 2.0  # band std > 2x the strip's median band std = edge line / selvedge stripe
JUMP_MIN = 15.0   # brightness jump between neighbouring bands > max(15, 5x median jump) = edge


def make_tiles(img, size=TILE):
    # Cut from left to right without overlap. Discard the incomplete rightmost tile.
    h, w = img.shape[:2]
    return [(i, img[:, x:x + size]) for i, x in enumerate(range(0, w - size + 1, size))]


def list_images(folder):
    # Include supported images in all subfolders, in a consistent order.
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))


def edge_tiles(img, mask=None, size=TILE):
    """Return one bool per tile: True if the tile contains background or the fabric edge.

    Looks at 8-px column bands of the whole strip. A band is 'edge' if it has saturated
    (background) pixels, much higher texture than the rest of the strip, or a sharp
    brightness jump to its neighbour. Bands at or next to a defect (mask) are not counted,
    so a defect is never mistaken for an edge.
    """
    # Inspect narrow vertical bands spanning the full image height.
    # Floating-point values preserve precision during brightness calculations.
    g = img.astype(np.float32)
    nb = g.shape[1] // BAND
    bands = [g[:, i * BAND:(i + 1) * BAND] for i in range(nb)]
    mean = np.array([b.mean() for b in bands])
    std = np.array([b.std() for b in bands])
    sat = np.array([((b >= 250) | (b <= 5)).mean() for b in bands]) >= SAT_FRAC

    # Estimate typical fabric texture from bands without near-black/white background.
    # Medians reduce the influence of unusually bright or textured bands.
    ok = ~sat
    med_std = np.median(std[ok]) if ok.any() else 1.0
    diff = np.abs(np.diff(mean))
    ok_d = ok[1:] & ok[:-1]
    med_d = np.median(diff[ok_d]) if ok_d.any() else 1.0
    big = (diff > max(JUMP_MIN, 5 * med_d)) & ok_d
    # Mark both sides of each unusually large brightness change.
    jump = np.zeros(nb, bool)
    jump[1:] |= big
    jump[:-1] |= big

    # Background, unusually strong texture, or a brightness jump can signal an edge.
    flag = sat | (std > STD_FACTOR * med_std) | jump
    if mask is not None:
        # Defects can resemble edges. Suppress texture/jump flags near defects,
        # while keeping background flags active.
        near_defect = np.array([(mask[:, i * BAND:(i + 1) * BAND] > 0).any() for i in range(nb)])
        near_defect = np.convolve(near_defect, np.ones(5), "same") > 0  # +-2 bands
        flag &= ~(near_defect & ~sat)

    # Flag a tile if any band inside it is flagged. Tile width must be a multiple
    # of BAND; the default settings give 32 bands per tile.
    per = size // BAND
    return [bool(flag[i * per:(i + 1) * per].any()) for i in range(g.shape[1] // size)]


# ---------- 0. start from a clean output folder (no stale tiles) ----------
# Remove old generated files before rebuilding the output folders.
for d in SUBDIRS:
    shutil.rmtree(OUT / d, ignore_errors=True)
    (OUT / d).mkdir(parents=True, exist_ok=True)

rows = []  # one row per tile -> tiles.csv


def add_row(name, src, i, folder, edge, defect_px):
    # Source names follow <image ID>_<defect code>_<fabric code>.
    # Save the source and position so each tile can be traced back.
    parts = src.stem.split("_")
    rows.append({"tile": name, "source": src.stem, "defect_code": parts[1], "fabric": parts[2],
                 "tile_idx": i, "folder": folder, "edge": int(edge), "defect_px": defect_px})


# ---------- audit: duplicate raw files ----------
hashes = defaultdict(list)
for p in list_images(RAW):
    hashes[hashlib.md5(p.read_bytes()).hexdigest()].append(p.relative_to(RAW).as_posix())
dups = [v for v in hashes.values() if len(v) > 1]
print("duplicate raw files:", dups if dups else "none")

# ---------- 1. defect-free images ----------
for p in list_images(RAW / "NODefect_images"):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    edges = edge_tiles(img)
    for i, t in make_tiles(img):
        name = f"{p.stem}_t{i:02d}.png"
        sub = "edge" if edges[i] else "good"
        cv2.imwrite(str(OUT / sub / name), t)
        add_row(name, p, i, sub, edges[i], 0)

# ---------- 2. defective images + masks ----------
# some images have several masks (e.g. 0044_019_04_mask1.png, _mask2.png) -> merge them
mask_lookup = defaultdict(list)
for p in list_images(RAW / "Mask_images"):
    mask_lookup[re.sub(r"_mask\d*$", "", p.stem)].append(p)

for p in list_images(RAW / "Defect_images"):
    mpaths = mask_lookup.get(p.stem)
    if not mpaths:
        print("no mask for", p.name)
        continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    msks = [cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) for m in mpaths]
    if any(m.shape != img.shape for m in msks):
        print("size mismatch", p.name, img.shape, [m.shape for m in msks])
        continue
    # Merge all masks: a pixel is defective if any mask exceeds the threshold.
    # The result contains only 0 (clean) and 255 (defect).
    msk = (np.maximum.reduce(msks) > MASK_THRESH).astype(np.uint8) * 255
    if not msk.any():
        # labelled defective but nothing marked: neither a defect example nor a trusted clean one
        print("empty mask, skipped", p.name)
        continue
    edges = edge_tiles(img, msk)
    k = 2 * CLEAN_MARGIN + 1
    grown = cv2.dilate(msk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # Slice the image and its mask at matching positions to keep labels aligned.
    for (i, t), (_, m), (_, g) in zip(make_tiles(img), make_tiles(msk), make_tiles(grown)):
        name = f"{p.stem}_t{i:02d}.png"
        # Even one labelled pixel makes this tile a defect tile.
        defect_px = int((m > 0).sum())
        if defect_px:
            sub = "defect"  # defect tiles always stay here; tiles.csv marks the ones that also touch an edge
            cv2.imwrite(str(OUT / "mask" / name), m)
        elif g.any():
            # no defect pixel, but within CLEAN_MARGIN px of one: not trusted as clean, not saved
            add_row(name, p, i, "near_defect", edges[i], 0)
            continue
        else:
            sub = "edge" if edges[i] else "defect_image_clean_tile"
        cv2.imwrite(str(OUT / sub / name), t)
        add_row(name, p, i, sub, edges[i], defect_px)

# ---------- 3. index + count ----------
# Write one index row per image tile; mask files have no separate rows.
# At least one tile is required because the first row supplies the column names.
with open(OUT / "tiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for d in SUBDIRS:
    print(d, len(list((OUT / d).glob("*.png"))))
print("near_defect (not saved):", sum(r["folder"] == "near_defect" for r in rows))
print("defect tiles that also touch an edge:", sum(r["folder"] == "defect" and r["edge"] for r in rows))

# ---------- 4. per-fabric audit: tiles and source strips per folder ----------
folders = ["good", "edge", "defect", "defect_image_clean_tile", "near_defect"]
print("\nfabric | " + " | ".join(f"{f} tiles/strips" for f in folders))
for fab in sorted({r["fabric"] for r in rows}):
    cells = []
    for f in folders:
        sel = [r for r in rows if r["fabric"] == fab and r["folder"] == f]
        cells.append(f"{len(sel)}/{len({r['source'] for r in sel})}")
    print(fab, "|", " | ".join(cells))
