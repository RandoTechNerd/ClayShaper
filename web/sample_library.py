"""
Web-build sample library: same API as the desktop version, but backed by the
small curated asset set bundled with the static site (assets/models + thumbs
are pre-rendered so the gallery is instant — no slicing on load). Thumbnails
for user uploads are still rendered live with the real slicer, exactly like
the desktop build.
"""

import base64
import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "assets", "models")
THUMB_DIR = os.path.join(HERE, "assets", "thumbs")
GCODE_DIR = os.path.join(HERE, "assets", "gcodes")
RENDER_CACHE = os.path.join(HERE, "sample_thumbs")   # runtime cache (uploads)

# name -> category for the bundled set; anything new dropped into assets/models
# shows up under "Classics".
_CATEGORIES = {
    "Bowl": "Classics",
    "Taco Bell Bag": "Classics",
    "Spiral Vase": "Vase",
    "Octopus Vase": "Vase",
}

BASE_COLOR = (33, 150, 243)
WALL_COLOR = (193, 124, 83)
BG_COLOR = (250, 248, 245)


def list_sample_stls():
    """[(display_name, category, path)] for the bundled STLs."""
    if not os.path.isdir(MODEL_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(MODEL_DIR)):
        if not fn.lower().endswith(".stl"):
            continue
        name = os.path.splitext(fn)[0].strip()
        out.append((name, _CATEGORIES.get(name, "Classics"),
                    os.path.join(MODEL_DIR, fn)))
    return out


def list_sample_gcodes():
    """[(display_name, path)] for the bundled factory G-codes."""
    if not os.path.isdir(GCODE_DIR):
        return []
    return [(os.path.splitext(fn)[0].strip(), os.path.join(GCODE_DIR, fn))
            for fn in sorted(os.listdir(GCODE_DIR))
            if fn.lower().endswith(".gcode")]


def _project_iso(x, y, z):
    sx = (x - y) * 0.866
    sy = z - (x + y) * 0.30
    return sx, sy


def _render_sliced_thumbnail(stl_path, size=360):
    """Coarse-slice the STL and draw its toolpath in isometric view (uploads)."""
    from stl_slicer import STLSlicer
    from clay_lib import PRINTER_PROFILES

    profile = PRINTER_PROFILES["Eazao Potter"]
    slicer = STLSlicer(stl_path, profile, nozzle=3.0, layer_height=2.0)
    layers = slicer.slice(bottom_layers=2, staggered=True, vase_mode=False,
                          path_resolution=3.0)
    if not layers:
        return None
    base = [l for l in layers if l["type"] == "bottom"]
    walls = [l for l in layers if l["type"] != "bottom"]
    if len(walls) > 45:
        walls = walls[::math.ceil(len(walls) / 45)]
    layers = base + walls

    polylines = []
    minx = miny = math.inf
    maxx = maxy = -math.inf
    cx, cy = profile["center_x"], profile["center_y"]
    for layer in layers:
        color = BASE_COLOR if layer["type"] == "bottom" else WALL_COLOR
        z = layer["z"]
        for path in layer["paths"]:
            if path is None:
                continue
            pts = []
            for px, py in path.coords:
                sx, sy = _project_iso(px - cx, py - cy, z)
                pts.append((sx, sy))
                minx, maxx = min(minx, sx), max(maxx, sx)
                miny, maxy = min(miny, sy), max(maxy, sy)
            if len(pts) >= 2:
                polylines.append((pts, color))
    if not polylines:
        return None

    pad = size * 0.10
    span = max(maxx - minx, maxy - miny, 1e-6)
    scale = (size - 2 * pad) / span
    ox = (size - (maxx - minx) * scale) / 2 - minx * scale
    oy = (size - (maxy - miny) * scale) / 2 - miny * scale
    img = Image.new("RGB", (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)
    for pts, color in polylines:
        screen = [(x * scale + ox, size - (y * scale + oy)) for x, y in pts]
        draw.line(screen, fill=color, width=2, joint="curve")
    return img


def get_thumbnail(stl_path):
    """Path to a thumbnail PNG: pre-rendered for bundled samples, rendered
    (and cached) live for anything else."""
    name = os.path.splitext(os.path.basename(stl_path))[0].strip()
    pre = os.path.join(THUMB_DIR, name + ".png")
    if os.path.exists(pre):
        return pre
    os.makedirs(RENDER_CACHE, exist_ok=True)
    cached = os.path.join(RENDER_CACHE, name.replace(" ", "_") + ".png")
    if os.path.exists(cached):
        return cached
    try:
        img = _render_sliced_thumbnail(stl_path)
    except Exception:
        return None
    if img is None:
        return None
    img.save(cached)
    return cached


def get_thumbnail_b64(stl_path):
    thumb = get_thumbnail(stl_path)
    if thumb is None:
        return None
    with open(thumb, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
