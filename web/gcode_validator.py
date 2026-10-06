"""
G-code validation for clay printing.

Parses a G-code program and checks it against a printer profile for things that
would ruin a print (or the machine):

  * clay safety  - the mandatory M302 / M163 / M164 start codes are present
  * bounds       - every move stays inside the build volume
  * speeds       - extruding feedrates stay within the machine's spec range,
                   and no move exceeds the mechanical hard cap
  * flow         - no local over-extrusion spikes, and the total clay volume
                   fits in the cartridge

It is deliberately dependency-free (pure Python + math) so it can run on any
G-code: ClayShaper's own output, sliced STLs, or a file the user pastes in.
"""

import math
import re

# Severity levels, worst first.
FAIL = "fail"        # will not print correctly / could damage the machine
WARN = "warn"        # likely a problem, worth a look ("caution")
SUGGEST = "suggest"  # prints fine, but could print better
INFO = "info"        # informational / passed

# A word is a letter and a number. The number may carry a sign and may start
# with the dot: PrusaSlicer, Orca and Bambu write "E.01637" and some tools
# "X+5". The old pattern needed a digit before the dot, so those words were
# dropped: a PrusaSlicer-style relative-E Coil Bowl read 0.0 ml of clay
# instead of 45.6, and its preview came out empty. Exponents stay OUT of the
# number on purpose, because Marlin itself stops reading a number at the
# letter E ("X10E5" is X10 then E5); scientific notation is flagged instead.
_TOKEN = re.compile(r"([A-Za-z])([-+]?(?:\d+\.?\d*|\.\d+))")
# Python's str(1e-05) style: a lowercase e straight after a digit on an
# uppercase line. Firmware reads "E1e-05" as E1, not 0.00001.
_SCI = re.compile(r"\d\.?\d*e[-+]?\d")

# Classic-jerk constants for the motion check, from Eazao's own Cura
# definition (eazao_potter.def.json: Z jerk 0.3 mm/s) and Marlin 1.1.9's
# MIN_STEPS_PER_SEGMENT 6 at 100 steps/mm on X/Y and 400 on Z: a move under
# 6 steps on every axis is dropped and folded into the next one, so those
# never brake on their own.
_Z_JERK = 0.3
_MERGE_XY = 0.06
_MERGE_Z = 0.015
# A junction counts as a hitch when it forces the head below 40% of the
# commanded speed, the same line the regression harness draws.
_HITCH_FRAC = 0.4


class Issue:
    def __init__(self, severity, category, message, line_no=None):
        self.severity = severity
        self.category = category
        self.message = message
        self.line_no = line_no

    def __repr__(self):
        loc = f" (line {self.line_no})" if self.line_no else ""
        return f"[{self.severity.upper()}] {self.category}: {self.message}{loc}"


class ValidationReport:
    def __init__(self, issues, stats):
        self.issues = issues
        self.stats = stats

    @property
    def ok(self):
        return not any(i.severity == FAIL for i in self.issues)

    @property
    def has_warnings(self):
        return any(i.severity == WARN for i in self.issues)

    @property
    def verdict(self):
        """One of: 'fail', 'caution', 'pass_suggest', 'pass'."""
        sevs = {i.severity for i in self.issues}
        if FAIL in sevs:
            return "fail"
        if WARN in sevs:
            return "caution"
        if SUGGEST in sevs:
            return "pass_suggest"
        return "pass"

    def by_severity(self, severity):
        return [i for i in self.issues if i.severity == severity]


def _parse_line(line):
    """Return dict of word -> float for a G-code line, ignoring comments."""
    code = line.split(";", 1)[0].strip()
    if not code:
        return None, {}
    words = dict(
        (m.group(1).upper(), float(m.group(2))) for m in _TOKEN.finditer(code)
    )
    verb = None
    if "G" in words:
        verb = "G%d" % int(words["G"])
    elif "M" in words:
        verb = "M%d" % int(words["M"])
    return verb, words


# Words that carry a length, so G20 (inches) scales them. M-code arguments
# such as M163's mixing ratio are not lengths and must not be touched.
_LENGTH_WORDS = "XYZEFIJR"


def _to_mm(words, unit):
    if unit == 1.0:
        return words
    return {k: (v * unit if k in _LENGTH_WORDS else v) for k, v in words.items()}


def _arc_chords(x, y, z, nx, ny, nz, de, w, clockwise, step=1.0):
    """
    A G2 (clockwise) / G3 arc from (x, y, z) to (nx, ny, nz) as chords of at
    most `step` mm, each (x, y, z, de) with Z and E spread evenly, the way
    Marlin segments it (MM_PER_ARC_SEGMENT 1 mm). Centre from I/J (relative
    to the start) or from R, using Marlin's own R formula (negative R = the
    long way round). Start == end with I/J is a full circle. Anything that
    does not describe an arc falls back to one straight move.
    """
    straight = ((nx, ny, nz, de),)
    if "I" in w or "J" in w:
        cx, cy = x + w.get("I", 0.0), y + w.get("J", 0.0)
    elif w.get("R"):
        r = w["R"]
        dx, dy = nx - x, ny - y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return straight
        sgn = -1.0 if (clockwise != (r < 0)) else 1.0
        h2 = (r - 0.5 * d) * (r + 0.5 * d)
        h = math.sqrt(h2) if h2 > 0 else 0.0
        cx = (x + nx) * 0.5 + sgn * h * (-dy / d)
        cy = (y + ny) * 0.5 + sgn * h * (dx / d)
    else:
        return straight
    rad = math.hypot(x - cx, y - cy)
    if rad < 1e-6:
        return straight
    a0 = math.atan2(y - cy, x - cx)
    a1 = math.atan2(ny - cy, nx - cx)
    sweep = (a0 - a1) if clockwise else (a1 - a0)
    sweep %= 2 * math.pi
    if sweep < 1e-9:
        sweep = 2 * math.pi      # back to the start: a full circle
    n = min(3600, max(1, int(math.ceil(rad * sweep / step))))
    out = []
    for k in range(1, n + 1):
        t = k / n
        a = a0 - sweep * t if clockwise else a0 + sweep * t
        if k == n:
            px, py = nx, ny
        else:
            px, py = cx + rad * math.cos(a), cy + rad * math.sin(a)
        out.append((px, py, z + (nz - z) * t, de / n))
    return out


def _cura_settings(text):
    """Cura's ;SETTING_3 block, re-joined. Cura splits it over many comment
    lines (22 in Eazao's Bowl.gcode), so a key can straddle two of them."""
    parts = [ln.strip()[len(";SETTING_3"):].lstrip()
             for ln in text.splitlines() if ln.lstrip().startswith(";SETTING_3")]
    return "".join(parts)


def _header_float(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def file_settings(text):
    """
    The print settings a file declares about itself, or None for each one it
    does not state:
      layer_height, base_layer_height, line_width, first_layer_height.

    ClayShaper writes ';Layer height:', ';Base layer height:' (only when it
    differs) and ';Line width:'. Cura writes ';Layer height:' and keeps the
    rest in its ;SETTING_3 block (layer_height_0, line_width). PrusaSlicer
    and Orca write '; first_layer_height = ' / '; initial_layer_print_height
    = ' in their config footer. A percentage first layer (Prusa "75%") is not
    read, so it can never make a check stricter than it should be.
    """
    cura = _cura_settings(text)
    out = {
        "layer_height": _header_float(r"^;Layer height:\s*([0-9.]+)", text),
        "base_layer_height": _header_float(r"^;Base layer height:\s*([0-9.]+)", text),
        "line_width": _header_float(r"^;Line width:\s*([0-9.]+)", text),
        "first_layer_height": _header_float(r"^;First layer height:\s*([0-9.]+)", text),
    }
    if cura:
        # Keys follow a literal "\n" in the escaped block, so anchor on that
        # (a plain \b would also match wall_line_width_0 and friends).
        if out["line_width"] is None:
            out["line_width"] = _header_float(r"(?:\\n|^)line_width\s*=\s*([0-9.]+)", cura)
        if out["first_layer_height"] is None:
            out["first_layer_height"] = _header_float(
                r"(?:\\n|^)layer_height_0\s*=\s*([0-9.]+)", cura)
    if out["first_layer_height"] is None:
        out["first_layer_height"] = _header_float(
            r"^;\s*(?:first_layer_height|initial_layer_print_height)\s*=\s*([0-9.]+)(?![0-9.]*%)",
            text)
    return out


def extract_toolpath(text, max_points=36000):
    """
    Parse a G-code program into plottable polylines for a 3D preview.

    Returns {"base": (xs, ys, zs), "wall": (xs, ys, zs)} where each list is a
    sequence of coordinates with None separators between discontinuous runs
    (travels). Extruding moves under a Cura ";TYPE:SKIN" section count as base;
    everything else extruding is wall. Long files are downsampled to keep the
    plot responsive.
    """
    absolute = True
    e_rel_mode = False       # M83; G91 also makes E relative (Marlin)
    unit = 1.0               # G20 inches -> 25.4
    x = y = z = 0.0
    e = 0.0
    cur_kind = "wall"
    pts = []   # (x, y, z, kind) for extruding moves; None marks a break

    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped.startswith(";TYPE:"):
            cur_kind = "base" if "SKIN" in stripped else "wall"
            continue
        verb, w = _parse_line(ln)
        if verb is None:
            continue
        if verb == "G90":
            absolute = True; continue
        if verb == "G91":
            absolute = False; continue
        if verb == "M82":
            e_rel_mode = False; continue
        if verb == "M83":
            e_rel_mode = True; continue
        if verb == "G20":
            unit = 25.4; continue
        if verb == "G21":
            unit = 1.0; continue
        if unit != 1.0:
            w = _to_mm(w, unit)
        if verb == "G92":
            if "X" in w: x = w["X"]
            if "Y" in w: y = w["Y"]
            if "Z" in w: z = w["Z"]
            if "E" in w: e = w["E"]
            continue
        if verb not in ("G0", "G1", "G2", "G3"):
            continue

        nx, ny, nz = x, y, z
        if "X" in w: nx = w["X"] if absolute else x + w["X"]
        if "Y" in w: ny = w["Y"] if absolute else y + w["Y"]
        if "Z" in w: nz = w["Z"] if absolute else z + w["Z"]

        e_abs = absolute and not e_rel_mode
        de = 0.0
        if "E" in w:
            de = (w["E"] - e) if e_abs else w["E"]
            e = w["E"] if e_abs else e + w["E"]

        if verb in ("G2", "G3"):
            subs = _arc_chords(x, y, z, nx, ny, nz, de, w, verb == "G2")
        else:
            subs = ((nx, ny, nz, de),)
        for nx, ny, nz, de in subs:
            if de > 1e-9 and (nx != x or ny != y or nz != z):
                if not pts or pts[-1] is None:
                    pts.append((x, y, z, cur_kind))   # run start: include the origin
                pts.append((nx, ny, nz, cur_kind))
            elif pts and pts[-1] is not None:
                pts.append(None)                       # travel: break the line
            x, y, z = nx, ny, nz

    # Downsample long paths. Every run keeps its first and last point (and
    # the points either side of a base/wall change): the old index stride
    # dropped run ends, so 1701 of the Bulldog's 2112 runs lost their tail,
    # up to 59.8 mm of path, and the preview showed gaps that are not in
    # the file. Interior points are still thinned by the stride.
    n_real = sum(1 for p in pts if p is not None)
    step = max(1, n_real // max_points)
    out = {"base": ([], [], []), "wall": ([], [], [])}
    i = 0
    prev_kind = None
    n_pts = len(pts)
    for j, p in enumerate(pts):
        if p is None:
            if prev_kind is not None:
                arr = out[prev_kind]
                arr[0].append(None); arr[1].append(None); arr[2].append(None)
            prev_kind = None
            continue
        i += 1
        px, py, pz, kind = p
        if step > 1 and (i % step) and prev_kind == kind:
            nxt = pts[j + 1] if j + 1 < n_pts else None
            if nxt is not None and nxt[3] == kind:
                continue
        if prev_kind is not None and kind != prev_kind:
            arr = out[prev_kind]
            arr[0].append(None); arr[1].append(None); arr[2].append(None)
        arr = out[kind]
        arr[0].append(px); arr[1].append(py); arr[2].append(pz)
        prev_kind = kind
    return out


_BASE_SEAM_MSG = (
    "Base rings stack directly on each other (no stagger). Offsetting "
    "alternate base layers half a line width spreads the seams and makes "
    "a stronger, more watertight bottom — enable “Staggered base” when slicing.")


def _analyze_geometry(segments, bed_x, bed_y, line_width=3.0, staggered=None):
    """
    Clay-specific geometric checks on the extrusion segments.

    segments: list of (x0, y0, x1, y1, layer_idx, is_skin, is_wall) extrusions.
    Returns a list of Issues:
      * unsupported extrusion ("printing in thin air") — each layer's centerline
        must land on the previous layer's clay (bead width + small tolerance);
      * large solid areas (drying/shrinkage cracks in clay);
      * stacked (unstaggered) base rings — suggestion only.
    """
    import numpy as np
    issues = []
    if not segments:
        return issues

    cell = 1.0  # mm grid
    gw, gh = int(bed_x / cell) + 3, int(bed_y / cell) + 3
    half_bead_cells = max(1, int(round(line_width / 2.0)))

    # Bucket segments per layer, keeping order. Per-layer tuples are
    # (x0, y0, x1, y1, is_skin, is_wall).
    layers = {}
    for x0, y0, x1, y1, li, skin, wall in segments:
        layers.setdefault(li, []).append((x0, y0, x1, y1, skin, wall))
    layer_ids = sorted(layers)

    def rasterize(segs):
        """Centerline occupancy grid + point list for a layer."""
        pts_all = []
        for x0, y0, x1, y1, *_ in segs:
            d = math.hypot(x1 - x0, y1 - y0)
            n = max(int(d / cell), 1)
            t = np.linspace(0.0, 1.0, n + 1)
            pts_all.append(np.column_stack((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)))
        pts = np.vstack(pts_all)
        ix = np.clip((pts[:, 0] / cell).astype(int), 0, gw - 1)
        iy = np.clip((pts[:, 1] / cell).astype(int), 0, gh - 1)
        grid = np.zeros((gw, gh), dtype=bool)
        grid[ix, iy] = True
        return grid, pts

    # numpy 4-neighbour dilation (see nearest.py) — matches scipy's default
    # structure and zero padding, without the scipy dependency.
    from nearest import binary_dilation

    def dilate(grid, iters):
        return binary_dilation(grid, iterations=iters)

    # --- 1. Support ("thin air") --------------------------------------------
    # A layer's centerline must fall on the previous layer's bead footprint
    # (previous centerline dilated by half a bead + 1 cell tolerance).
    #
    # Severity is position-aware, calibrated on Eazao's factory files:
    #  * The killer failure is at the BASE->WALL seam (first wall layers not
    #    landing on the base): everything above depends on it -> FAIL.
    #    (A real failed print showed 60% thin-air on the first wall layer.)
    #  * Mid-wall/high overhangs are how sculptural pieces (drapes, chins)
    #    print — the factory Draped Vase measures up to 74% layer-over-layer
    #    shift at its folds and prints fine -> CAUTION only.
    support_iters = half_bead_cells + 1
    last_skin = max((li for li in layer_ids if any(s[4] for s in layers[li])),
                    default=None)
    seam_bad = []                # (layer, frac) near the base->wall seam
    wall_worst = (0.0, None)     # worst mid-wall overhang
    prev_support = None
    for li in layer_ids:
        grid, pts = rasterize(layers[li])
        if prev_support is not None:
            ix = np.clip((pts[:, 0] / cell).astype(int), 0, gw - 1)
            iy = np.clip((pts[:, 1] / cell).astype(int), 0, gh - 1)
            frac = float((~prev_support[ix, iy]).mean())
            at_seam = last_skin is not None and last_skin < li <= last_skin + 2
            if at_seam and frac > 0.30:
                seam_bad.append((li, frac))
            elif frac > wall_worst[0]:
                wall_worst = (frac, li)
        prev_support = dilate(grid, support_iters)

    if seam_bad:
        li, frac = max(seam_bad, key=lambda t: t[1])
        issues.append(Issue(FAIL, "Support",
            f"The wall doesn't land on the base: layer {li} (first wall layers) has "
            f"{frac*100:.0f}% of its path printing in thin air past the base's edge. "
            f"The wall will collapse. Re-slice this model (older slicer files had a "
            f"wall/base alignment bug) or widen the base."))
    if wall_worst[0] > 0.20:
        frac, li = wall_worst
        issues.append(Issue(WARN, "Support",
            f"Layer {li} shifts {frac*100:.0f}% of its path off the layer below — a "
            f"steep overhang or drape. Sculptural pieces can print this, but watch "
            f"for sagging; stiffer clay helps."))

    # --- 2. Large solid areas (SKIN layers) ----------------------------------
    solid_area_cm2 = 0.0
    solid_layers = 0
    for li in layer_ids:
        segs = layers[li]
        if not any(s[4] for s in segs):
            continue
        grid, _ = rasterize([s for s in segs if s[4]])
        footprint = dilate(grid, half_bead_cells)
        area = footprint.sum() * cell * cell / 100.0   # cm^2
        if area > solid_area_cm2:
            solid_area_cm2 = area
        solid_layers += 1
    if solid_area_cm2 > 130:
        issues.append(Issue(WARN, "Solid area",
            f"A solid layer covers ~{solid_area_cm2:.0f} cm². Large unbroken clay "
            f"slabs dry unevenly and crack — consider a smaller footprint or an "
            f"open/patterned base."))
    elif solid_area_cm2 > 80:
        issues.append(Issue(SUGGEST, "Solid area",
            f"The solid base covers ~{solid_area_cm2:.0f} cm². Watch drying — big "
            f"solid areas can crack; slower drying or a patterned base helps."))

    # --- 2a. Path loops (hairpin reversals) -----------------------------------
    # A path that doubles back on itself >150° makes the nozzle loop over its
    # own bead — it drags and tears the clay, leaving holes (confirmed on a
    # real print from an older slicer: 193 loop-backs, hole at the loop spot).
    # Well-formed files (all 19 factory G-codes, current Studio output) have ~0.
    # Only count reversals that RETRACE the previous segment (the new segment's
    # end lands back on the line just printed). Legit sharp corners and infill
    # U-turns reverse direction but diverge into new territory — those are fine.
    def _pt_seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        if L2 < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
        return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

    # WALL sections only: Cura's skin/infill patterns retrace by design, and
    # thin decorative wall features (e.g. Berry Pot's bumps) measure up to ~63
    # legit retraces — the broken-slicer failure measures 250+. Threshold 100.
    hairpin_count = 0
    hairpin_layer = None
    for li in layer_ids:
        segs = layers[li]
        prev = None
        layer_hp = 0
        for s in segs:
            if not s[5]:          # non-wall (skin/infill): reset and skip
                prev = None
                continue
            x0, y0, x1, y1 = s[:4]
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy)
            if ln < 0.05:
                continue
            if prev is not None:
                px0, py0, px1, py1, pdx, pdy = prev
                connected = abs(x0 - px1) < 0.01 and abs(y0 - py1) < 0.01
                if connected and (dx * pdx + dy * pdy) / (ln * math.hypot(pdx, pdy)) < -0.85:
                    if _pt_seg_dist(x1, y1, px0, py0, px1, py1) < 0.4:
                        layer_hp += 1
            prev = (x0, y0, x1, y1, dx, dy)
        hairpin_count += layer_hp
        if layer_hp and hairpin_layer is None:
            hairpin_layer = li
    if hairpin_count >= 100:
        issues.append(Issue(WARN, "Path loops",
            f"The extrusion path doubles back on itself {hairpin_count} times "
            f"(first at layer {hairpin_layer}). The nozzle loops over its own "
            f"bead there — clay drags and tears, leaving holes. This is typical "
            f"of files from older slicers; re-slice the model in ClayShaper."))

    # --- 2b. Surface flicker (lone-layer zig-zag) -----------------------------
    # A coil that deviates >0.8mm from BOTH its vertical neighbors, with both
    # neighbors on the SAME side, is a lone in/out zig-zag (unstable slicing at
    # folds): it prints as rough terraces of exposed coil ends, yet every step
    # is small enough to pass the support check. Smooth drift is different —
    # there the neighbors sit on opposite sides and cancel.
    # Calibrated on the Eazao corpus + a known-bad slice: measure point-to-PATH
    # (densified) distance, single-loop (vase-style) layers only — multi-island
    # bodies (cartoon models) make nearest-neighbor sides meaningless — and trim
    # the first/last triples (spiral ramp-in/out reads as fake deviation).
    # numpy stand-in for cKDTree (see nearest.py) — keeps scipy out of the
    # browser build; the check now runs everywhere instead of silently
    # degrading when scipy is unavailable.
    from nearest import NearestPoints as cKDTree
    wall_ids = [li for li in layer_ids if not any(s[4] for s in layers[li])
                and len(layers[li]) > 15]
    if cKDTree is not None and len(wall_ids) >= 8:
        def runs_in(segs):
            n = 1
            for i in range(1, len(segs)):
                if (abs(segs[i][0] - segs[i-1][2]) > 0.01
                        or abs(segs[i][1] - segs[i-1][3]) > 0.01):
                    n += 1
            return n

        single = sum(1 for li in wall_ids if runs_in(layers[li]) == 1)
        if single >= 0.9 * len(wall_ids):
            def densify_layer(segs, step=1.0):
                out = []
                for x0, y0, x1, y1, *_ in segs:
                    d = math.hypot(x1 - x0, y1 - y0)
                    n = max(int(d / step), 1)
                    t = np.linspace(0.0, 1.0, n + 1)
                    out.append(np.column_stack((x0 + (x1 - x0) * t,
                                                y0 + (y1 - y0) * t)))
                return np.vstack(out)

            dense = {li: densify_layer(layers[li]) for li in wall_ids}
            trees = {li: cKDTree(dense[li]) for li in wall_ids}
            fr = []
            for a, b, c in list(zip(wall_ids, wall_ids[1:], wall_ids[2:]))[2:-2]:
                cur = dense[b]
                if len(cur) > 400:
                    cur = cur[np.linspace(0, len(cur) - 1, 400).astype(int)]
                dp, ip = trees[a].query(cur)
                dn, ic = trees[c].query(cur)
                vp = cur - dense[a][ip]
                vn = cur - dense[c][ic]
                same_side = np.einsum("ij,ij->i", vp, vn) > 0
                fr.append(((same_side & (dp > 1.2) & (dn > 1.2)).mean(), b))
            if fr:
                fracs = np.asarray([f for f, _ in fr])
                n_bad = int((fracs > 0.15).sum())
                worst = float(fracs.max())
                if (n_bad >= 1 and worst >= 0.25) or n_bad >= 5:
                    wl = fr[int(np.argmax(fracs))][1]
                    issues.append(Issue(SUGGEST, "Surface flicker",
                        f"{max(n_bad,1)} wall layer(s) deviate >1.2 mm from BOTH "
                        f"their vertical neighbors (worst near layer {wl}, "
                        f"{worst*100:.0f}% of its path) — a layer-to-layer zig-zag "
                        f"or very sharp drape. It prints, but expect rough stepped "
                        f"texture there; if this is a ClayShaper slice, re-slicing with "
                        f"current contour smoothing usually clears it."))

    # --- 3. Stacked base rings (stagger off) ---------------------------------
    skin_ids = [li for li in layer_ids if any(s[4] for s in layers[li])]
    if staggered is not None:
        # The file states the setting outright, so use it and skip the
        # heuristic below, which only reads true on a round base: it
        # compares ring radii between layers, and on an irregular
        # footprint the radius varies more within a single ring than the
        # stagger moves it between layers. That misreads both ways — it
        # told people to switch on a stagger that was already on, and
        # stayed quiet on flared models where it was genuinely off.
        if not staggered and len(skin_ids) >= 2:
            issues.append(Issue(SUGGEST, "Base seams", _BASE_SEAM_MSG))
        skin_ids = []
    if len(skin_ids) >= 2:
        maxr = {}
        for li in skin_ids:
            pts_all = []
            for x0, y0, x1, y1, skin, *_ in layers[li]:
                if skin:
                    pts_all.append((x0, y0)); pts_all.append((x1, y1))
            p = np.asarray(pts_all)
            c = p.mean(axis=0)
            r_all = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
            # Judge the stagger on the FIRST INTERIOR ring, not the outermost.
            # The outer ring deliberately follows the model on every base layer
            # (staggering it would leave the next layer's rim unsupported), so
            # comparing outer radii now reports "no stagger" even when it is on.
            r_max = float(r_all.max())
            inner = r_all[r_all < r_max - 0.6 * line_width]
            maxr[li] = float(inner.max()) if inner.size else r_max
        # "Stacked" = ring shift much smaller than a stagger step (half a line
        # width); natural flare of a sloped wall is ~0.3-0.5mm and still counts.
        stack_tol = 0.35 * line_width
        pairs = list(zip(skin_ids, skin_ids[1:]))
        stacked = sum(1 for a, b in pairs if abs(maxr[a] - maxr[b]) < stack_tol)
        if pairs and stacked == len(pairs):
            issues.append(Issue(SUGGEST, "Base seams", _BASE_SEAM_MSG))
    return issues


def validate_gcode(text, profile, nozzle=None, layer_height=None,
                   max_spike_examples=5, use_file_settings=False):
    """
    Validate a G-code string against a printer profile.

    nozzle / layer_height are optional; when given, the flow check compares the
    file's real extrusion against the volume it *should* be laying down.

    use_file_settings: when True, a bead width and layer height the file
    declares about itself (ClayShaper ';Line width:' / ';Layer height:',
    Cura's line_width) replace the nozzle / layer_height passed in. Validate
    mode wants this: a file sliced for a 1.6 mm nozzle checked against the
    sidebar's 3.0 mm read as "0.5x the expected flow" although it was right.
    Off by default, so callers that pass the real settings are unaffected.
    """
    issues = []
    lines = text.splitlines()

    bed_x = profile["bed_x"]
    bed_y = profile["bed_y"]
    max_z = profile["max_z"]
    max_feed = profile.get("max_feedrate", 3600)
    max_print = profile.get("max_print_speed", 2400)
    min_print = profile.get("min_print_speed", 0)
    # The machine's Z feed limit. Eazao's Cura definition caps Z at 5 mm/s,
    # which is the profile's z_speed (F300).
    z_max_mm_s = profile.get("z_speed", 300) / 60.0
    filament_area = math.pi * (profile.get("filament_dia", 1.75) / 2.0) ** 2
    cartridge_ml = profile.get("cartridge_ml")

    declared = file_settings(text)
    lw_from_file = False
    if use_file_settings:
        if declared["line_width"]:
            nozzle = declared["line_width"]
            lw_from_file = True
        if declared["layer_height"]:
            layer_height = declared["layer_height"]

    # --- 1. Clay safety start codes -----------------------------------------
    have = {"M302": False, "M163": False, "M164": False}
    for ln in lines:
        verb, _ = _parse_line(ln)
        if verb in have:
            have[verb] = True
    if not have["M302"]:
        issues.append(Issue(FAIL, "Clay safety",
            "Missing M302 (cold-extrusion enable). The machine has no heater and "
            "will refuse to extrude without it."))
    # Mixing-ratio codes only matter on a mixing hotend (Eazao). A profile whose
    # own start block never sends them (Tronxy, generic Marlin) must not be
    # told they are missing.
    needs_mix = "M163" in (profile.get("start_gcode") or "M163")
    if needs_mix and (not have["M163"] or not have["M164"]):
        issues.append(Issue(WARN, "Clay safety",
            "Missing M163/M164 mixing-ratio commands. Eazao's dual-material hotend "
            "expects them; extrusion may be wrong or blocked."))

    # --- 2. Walk the toolpath -----------------------------------------------
    absolute = True          # G90 default
    x = y = z = 0.0
    e = 0.0
    # E is relative under M83, and also under G91: Marlin's G91 switches
    # every axis to relative, E included, and G90 hands E back to whatever
    # M82/M83 last said. Ignoring that turned a "G91 / G1 Z2 E-0.5 / G1 Z-2
    # E0.5 / G90" lift into a 6925 E/mm spike and doubled the clay total.
    e_rel_mode = False       # M82 default (absolute E)
    unit = 1.0               # G21 mm; G20 inches multiplies lengths by 25.4
    inch_line = None
    feed = 0.0
    have_pos = False

    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]
    oob_count = 0
    oob_example = None
    lift_moves = []          # (line, z): Z-only overshoot by a non-extruding move
    overspeed_count = 0
    overspeed_example = None
    hardcap_hit = False
    hardcap_example = None

    e_per_mm = []            # (value, line_no) for extruding moves
    total_extruded = 0.0     # net positive E laid down (mm of "filament")
    z_values = set()
    layer_count_comment = None   # authoritative ;LAYER_COUNT: from Cura, if present
    cur_layer = None             # from ;LAYER: markers (Cura + our slicer)
    cur_skin = False             # from ;TYPE:SKIN sections
    cur_wall = True              # from ;TYPE:WALL* (files without TYPE count as wall)
    segments = []                # (x0, y0, x1, y1, layer_idx, is_skin, is_wall)

    first_ext = None             # (z, line) of the first extruding move
    last_ext_line = 0
    ext_z_min = math.inf
    ext_z_max = -math.inf
    low_z_count = 0
    low_z_example = None
    ext_len = 0.0                # extruding XY length
    slow_len = 0.0               # ... of it below the machine's minimum speed
    slow_example = None
    sci_count = 0
    sci_example = None

    # Motion check state (see section 4b).
    pend = None                  # start of a block still being merged
    prev_blk = None              # last merged block in this run
    prev_rev = False             # did the junction before prev_blk reverse?
    prev2_xy = None              # XY length of the block before prev_blk
    z_hitch = 0
    z_hitch_line = None
    retrace = 0
    retrace_line = None
    z_fast = 0
    z_fast_line = None

    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()
        if stripped.startswith(";"):
            if "LAYER_COUNT" in stripped:
                m = re.search(r"LAYER_COUNT:\s*(\d+)", stripped)
                if m:
                    layer_count_comment = int(m.group(1))
            elif stripped.startswith(";LAYER:"):
                try:
                    cur_layer = int(stripped.split(":")[1])
                except ValueError:
                    pass
            elif stripped.startswith(";TYPE:"):
                cur_skin = "SKIN" in stripped
                cur_wall = "WALL" in stripped
        verb, w = _parse_line(ln)
        if verb is None:
            continue
        if verb == "G90":
            absolute = True; continue
        if verb == "G91":
            absolute = False; continue
        if verb == "M82":
            e_rel_mode = False; continue
        if verb == "M83":
            e_rel_mode = True; continue
        if verb == "G20":
            unit = 25.4
            if inch_line is None:
                inch_line = i
            continue
        if verb == "G21":
            unit = 1.0; continue
        if verb in ("G0", "G1", "G2", "G3", "G92"):
            if "e" in stripped:
                code = stripped.split(";", 1)[0]
                if code[:1].isupper() and _SCI.search(code):
                    sci_count += 1
                    if sci_example is None:
                        sci_example = (i, code.strip()[:40])
            if unit != 1.0:
                w = _to_mm(w, unit)
        if verb == "G92":
            # Reset logical position (does not move). Update tracked coords.
            if "X" in w: x = w["X"]
            if "Y" in w: y = w["Y"]
            if "Z" in w: z = w["Z"]
            if "E" in w: e = w["E"]
            continue
        if verb not in ("G0", "G1", "G2", "G3"):
            continue

        if "F" in w:
            feed = w["F"]

        nx, ny, nz = x, y, z
        if "X" in w: nx = w["X"] if absolute else x + w["X"]
        if "Y" in w: ny = w["Y"] if absolute else y + w["Y"]
        if "Z" in w: nz = w["Z"] if absolute else z + w["Z"]

        # Extrusion delta
        de_all = 0.0
        if "E" in w:
            if absolute and not e_rel_mode:
                de_all = w["E"] - e
                e = w["E"]
            else:
                de_all = w["E"]
                e += w["E"]

        if "X" in w or "Y" in w:
            have_pos = True
        if verb in ("G2", "G3"):
            # Arcs are walked as chords of 1 mm or less, so their clay, bounds,
            # flow and support are checked like any other move. Ignoring them
            # passed a 60-circle vase as "0 ml, 0 layers" and gave an
            # arc-welded factory Bowl a false 65 E/mm spike on the next G1.
            subs = _arc_chords(x, y, z, nx, ny, nz, de_all, w, verb == "G2")
        else:
            subs = ((nx, ny, nz, de_all),)

        for nx, ny, nz, de in subs:
            moved = (nx != x) or (ny != y) or (nz != z)
            dist_xy = math.hypot(nx - x, ny - y)
            is_extrude = de > 1e-9 and dist_xy > 1e-6

            # Bounds (only meaningful once we have a real position; skip pure Z
            # homing dance which uses machine coordinates before first XY move).
            if have_pos and moved:
                for axis, val in ((0, nx), (1, ny), (2, nz)):
                    min_xyz[axis] = min(min_xyz[axis], val)
                    max_xyz[axis] = max(max_xyz[axis], val)
                # small tolerance for float noise / seam
                tol = 0.5
                xy_out = nx < -tol or nx > bed_x + tol or ny < -tol or ny > bed_y + tol
                z_high = nz > max_z + tol
                # Extrusion below the bed has its own, tighter check (3b): this
                # 0.5 mm tolerance let Design's no-base walls print at Z -0.29
                # with a PASS.
                z_low = nz < -tol and not is_extrude
                if z_high and not xy_out and not is_extrude:
                    # A lift that only overshoots the top may be the end-of-print
                    # move; decided after the walk, once the last extrusion is known.
                    lift_moves.append((i, nz))
                elif xy_out or z_high or z_low:
                    oob_count += 1
                    if oob_example is None:
                        oob_example = (i, round(nx, 1), round(ny, 1), round(nz, 1))

            # Speeds
            if feed > max_feed + 1:
                hardcap_hit = True
                if hardcap_example is None:
                    hardcap_example = (i, feed)
            elif is_extrude and feed > max_print + 1:
                overspeed_count += 1
                if overspeed_example is None:
                    overspeed_example = (i, feed)

            # Flow
            if is_extrude:
                e_per_mm.append((de / dist_xy, i))
                total_extruded += de
                z_values.add(round(nz, 3))
                # Layer id: prefer ;LAYER: markers; fall back to a z bin so plain
                # files without comments still get the geometric checks.
                li = cur_layer if cur_layer is not None else int(nz / (layer_height or 1.0))
                segments.append((x, y, nx, ny, li, cur_skin, cur_wall))

                if first_ext is None:
                    first_ext = (nz, i)
                last_ext_line = i
                ext_z_min = min(ext_z_min, nz)
                ext_z_max = max(ext_z_max, nz)
                if nz < 0.05:
                    low_z_count += 1
                    if low_z_example is None:
                        low_z_example = (i, nz)
                ext_len += dist_xy
                if min_print and 0 < feed < min_print - 1:
                    slow_len += dist_xy
                    if slow_example is None:
                        slow_example = (i, feed)

                # --- motion: per-move Z speed (no merging) ---
                dz = nz - z
                uz = abs(dz) / math.sqrt(dist_xy * dist_xy + dz * dz)
                if feed > 0 and uz > 0 and feed / 60.0 * uz > z_max_mm_s \
                        and z_max_mm_s / uz < _HITCH_FRAC * feed / 60.0:
                    z_fast += 1
                    if z_fast_line is None:
                        z_fast_line = i

                # --- motion: merged blocks (Marlin MIN_STEPS) ---
                if pend is None:
                    pend = (x, y, z)
                bx, by, bz = nx - pend[0], ny - pend[1], nz - pend[2]
                if abs(bx) >= _MERGE_XY or abs(by) >= _MERGE_XY or abs(bz) >= _MERGE_Z:
                    bl = math.sqrt(bx * bx + by * by + bz * bz)
                    bxy = math.hypot(bx, by)
                    if prev_blk is not None:
                        pbx, pby, pbz, pbl, pf, pbxy = prev_blk
                        # Classic jerk: the head may change its Z velocity by at
                        # most the Z jerk at a junction, so a change in slope dz/L
                        # caps the junction speed at Z_JERK / |change|.
                        dv = abs(bz / bl - pbz / pbl)
                        if dv * _HITCH_FRAC * min(feed, pf) / 60.0 > _Z_JERK:
                            z_hitch += 1
                            if z_hitch_line is None:
                                z_hitch_line = i
                        rev = (bxy > 1e-9 and pbxy > 1e-9
                               and (bx * pbx + by * pby) / (bxy * pbxy) < -0.99939)
                        # Out-and-back: two reversals of 178+ deg around a
                        # short block, with all three blocks the same length
                        # (within 10%), i.e. the nozzle runs the same line
                        # three times. The seam hairpins measure 0.743 /
                        # 0.743 / 0.744 mm; the factory Berry Pot's sharpest
                        # bumps (1.18 / 0.54 / 0.31) reverse just as hard but
                        # head somewhere new, so they are not counted.
                        if (rev and prev_rev and pbxy < 3.0 and prev2_xy is not None
                                and abs(prev2_xy - pbxy) <= 0.1 * pbxy
                                and abs(bxy - pbxy) <= 0.1 * pbxy):
                            retrace += 1
                            if retrace_line is None:
                                retrace_line = i
                        prev_rev = rev
                        prev2_xy = pbxy
                    prev_blk = (bx, by, bz, bl, feed, bxy)
                    pend = None
            elif moved or de < -1e-9 or de > 1e-9:
                # Travel, retract or a dab in place: the head stops, so the next
                # extrusion starts a fresh run.
                pend = prev_blk = prev2_xy = None
                prev_rev = False

            x, y, z = nx, ny, nz

    # --- 2b. Nothing to print ------------------------------------------------
    # A file with no extruding move is not "ready to print": the inch-unit
    # cup sliced to an empty toolpath and still read PASS.
    if first_ext is None:
        issues.append(Issue(FAIL, "Empty",
            "This file prints nothing: it has no extruding moves. If it came "
            "from a model, the model may be in inches or metres (far too small "
            "or too big), or it may not be closed."))

    # --- 3. Bounds verdict ---------------------------------------------------
    end_lift = [(li, lz) for li, lz in lift_moves if li > last_ext_line]
    for li, lz in lift_moves:
        if li <= last_ext_line:
            oob_count += 1
            if oob_example is None or li < oob_example[0]:
                oob_example = (li, None, None, round(lz, 1))
    if oob_count:
        li, ox, oy, oz = oob_example
        where = f"X{ox} Y{oy} Z{oz}" if ox is not None else f"Z{oz}"
        issues.append(Issue(FAIL, "Bounds",
            f"{oob_count} move(s) leave the {bed_x:g}x{bed_y:g}x{max_z:g} build "
            f"volume, e.g. {where}. Re-center or scale the model.", li))
    if end_lift:
        # Tronxy's end block lifts 10 mm in relative mode, so a pot that
        # tops out within 10 mm of the limit read FAIL although the print
        # itself fits. Marlin's software endstops stop the lift at the top.
        li, lz = max(end_lift, key=lambda t: t[1])
        issues.append(Issue(WARN, "Bounds",
            f"The lift after the print goes to Z{lz:.1f}, above the {max_z:g} mm "
            f"limit. The print itself fits, and the machine stops the lift at "
            f"its top, but make sure the nozzle clears the piece.", li))

    # --- 3b. First layer -----------------------------------------------------
    if low_z_count:
        li, lz = low_z_example
        issues.append(Issue(FAIL, "First layer",
            f"{low_z_count} extruding move(s) put the nozzle at Z{lz:.2f} mm or "
            f"lower, on or into the bed. The nozzle will scrape the bed and "
            f"block the clay. Raise the first layer.", li))
    if first_ext is not None:
        fz, fl = first_ext
        thick = max(declared["layer_height"] or layer_height or 0.0,
                    declared["base_layer_height"] or 0.0)
        if not thick:
            thick = (nozzle or 3.0) * 0.5
        # ClayShaper caps the first layer at 1.5x the base layer height
        # (0.6 mm floor), and an STL slice always states the base height
        # when it differs from the walls, so its first bead can never sit
        # higher than that. Eazao's Pleated Vase sliced at defaults starts
        # at Z3.60 against a 1.5 mm cap. Design files and plain files state
        # no first layer, so for them this is only a warning.
        cap = max(1.5 * thick, 0.6)
        clay_slice = re.search(r"^;Generated by ClayShaper \(STL slice\)", text,
                               re.MULTILINE) is not None
        if declared["first_layer_height"]:
            limit, sev = declared["first_layer_height"] + 0.5, FAIL
        elif clay_slice:
            limit, sev = cap + 0.5, FAIL
        else:
            limit, sev = cap + 0.5, WARN
        if fz > limit:
            issues.append(Issue(sev, "First layer",
                f"The first clay comes out at Z{fz:.2f} mm, well above the bed "
                f"(expected about {limit - 0.5:.2f} mm). The bead falls through "
                f"the air and will not stick. The model probably does not rest "
                f"flat, or its foot is thinner than the bead.", fl))

    # --- 4. Speed verdict ----------------------------------------------------
    if hardcap_hit:
        li, f = hardcap_example
        issues.append(Issue(FAIL, "Speed",
            f"Feedrate F{f:.0f} exceeds the machine hard cap of "
            f"F{max_feed:.0f} ({max_feed/60:.0f} mm/s).", li))
    if overspeed_count:
        li, f = overspeed_example
        issues.append(Issue(WARN, "Speed",
            f"{overspeed_count} extruding move(s) above the spec print speed "
            f"F{max_print:.0f} ({max_print/60:.0f} mm/s), e.g. F{f:.0f}. Clay may "
            f"tear or under-extrude.", li))
    # Too slow is a problem too: Design's handle copies 2-4 printed at the
    # modal F300 (5 mm/s against a 10 mm/s minimum). A few slow moves are
    # normal (the factory files have at most 0.04% of their path below it),
    # so this needs 5% of the extruded length.
    if ext_len > 0 and slow_len > 0.05 * ext_len:
        li, f = slow_example
        issues.append(Issue(WARN, "Speed",
            f"{slow_len / ext_len * 100:.0f}% of the path extrudes below the "
            f"machine's minimum print speed F{min_print:.0f} "
            f"({min_print/60:.0f} mm/s), e.g. F{f:.0f}. Clay pushed this slowly "
            f"builds pressure and blobs.", li))

    # --- 4b. Motion smoothness -----------------------------------------------
    # Three ways a toolpath forces the head to brake that a smooth file never
    # does. All 19 of Eazao's factory files score 0 on each, Pleated Vase
    # included. Plain turn angles are NOT counted: Pleated Vase brakes at
    # thousands of real creases on a junction-deviation planner and prints
    # fine, so an angle count cannot tell a defect from a design.
    # The v1.0.12 spiral seam scored (Z / retrace / steep): Twist Pot
    # 36 / 1 / 44, Eazao Bowl 16 / 0 / 0, Coil Bowl 4 / 7 / 2.
    #  * Z hitches: classic jerk (Eazao's firmware), after Marlin's short-move
    #    merge. Z stepped per point instead of per mm makes the slope change
    #    at every short segment, and a 0.3 mm/s Z jerk then caps the junction.
    #  * Retraces: the nozzle doubles back over the line it just laid and out
    #    again. Every planner stops twice there (Klipper and junction
    #    deviation included), and the clay is laid three times.
    #  * Steep moves: a move that climbs faster than the Z axis can go has to
    #    slow right down. Marlin folds the tiny ones into the next move, but
    #    Klipper and RepRapFirmware do not.
    # 1-2 events are noise; WARN from 20, the plan's max(20, 3x factory max).
    motion = z_hitch + retrace + z_fast
    if motion >= 3:
        first = min(l for l in (z_hitch_line, retrace_line, z_fast_line) if l)
        parts = []
        if z_hitch:
            parts.append(f"{z_hitch} sudden Z-speed change(s)")
        if retrace:
            parts.append(f"{retrace} out-and-back retrace(s)")
        if z_fast:
            parts.append(f"{z_fast} move(s) climbing faster than the Z axis can go")
        issues.append(Issue(WARN if motion >= 20 else SUGGEST, "Motion",
            f"The path makes the printer brake hard {motion} time(s): "
            f"{', '.join(parts)}. Expect stutter and small blobs there. "
            f"Eazao's own files have none; re-slicing in the current ClayShaper "
            f"usually clears it.", first))

    # --- 5. Flow verdict -----------------------------------------------------
    median_epm = None
    if e_per_mm:
        vals = sorted(v for v, _ in e_per_mm)
        median_epm = vals[len(vals) // 2]

        # Local spikes: extrusion far above the file's own norm = a blob/over-extrusion.
        spike_thresh = max(median_epm * 3.0, 0.05)
        spikes = [(v, li) for v, li in e_per_mm if v > spike_thresh]
        if spikes:
            worst = max(spikes)
            issues.append(Issue(WARN, "Flow",
                f"{len(spikes)} segment(s) extrude >3x the typical rate "
                f"({median_epm:.2f} E/mm), peaking at {worst[0]:.2f} E/mm. Likely "
                f"over-extrusion blobs.", worst[1]))

        # Expected flow, if we know nozzle + layer height.
        if nozzle and layer_height:
            expected = (layer_height * nozzle) / filament_area
            # The solid base may deliberately print thicker than the walls:
            # thin layers help walls hold overhangs but starve the base. Those
            # layers really do lay more clay per mm, and on a wide footprint
            # they are easily a tenth of the whole path, so without this the
            # sustained-flow check reads the base as over-extrusion. The
            # slicer writes the base height into the header when it differs.
            _bm = re.search(r";Base layer height:\s*([0-9.]+)", text)
            hot = expected
            if _bm:
                try:
                    hot = max(expected, (float(_bm.group(1)) * nozzle) / filament_area)
                except ValueError:
                    pass
            if expected > 0:
                ratio = median_epm / expected
                if ratio > 1.6 or ratio < 0.55:
                    issues.append(Issue(WARN, "Flow",
                        f"Typical flow {median_epm:.2f} E/mm is {ratio:.1f}x the "
                        f"expected {expected:.2f} E/mm for a {nozzle:g}mm bead at "
                        f"{layer_height:g}mm layers. Check flow / line width."))
                else:
                    # Sustained (not just spiky) over-extrusion: the top decile
                    # running hot means whole regions lay down too much clay.
                    p90 = vals[int(len(vals) * 0.9)]
                    if p90 > hot * 1.5:
                        issues.append(Issue(WARN, "Flow",
                            f"10% of the path extrudes at {p90:.2f} E/mm — over 1.5x "
                            f"the expected {hot:.2f}. Sustained over-extrusion "
                            f"blobs and drags in clay; check first-layer/flow settings."))

    # --- 5b. Geometry: support ("thin air"), solid areas, stagger -------------
    try:
        lw = nozzle if nozzle else 3.0
        _sm = re.search(r";Staggered base:\s*([01])", text)
        issues.extend(_analyze_geometry(
            segments, bed_x, bed_y, line_width=lw,
            staggered=(_sm.group(1) == "1") if _sm else None))
    except Exception:
        pass   # geometry analysis is best-effort; never block validation on it

    # --- 5c. Things worth a second look in how the file is written ----------
    if inch_line is not None:
        issues.append(Issue(WARN, "Units",
            "This file switches to inches (G20). The check converts it to mm, "
            "but clay slicers work in mm, so make sure that is what you meant.",
            inch_line))
    if sci_count:
        li, ex = sci_example
        issues.append(Issue(WARN, "Number format",
            f"{sci_count} line(s) write a number in scientific notation, e.g. "
            f"'{ex}'. Printer firmware stops reading at the e, so it gets a "
            f"different number than the file meant.", li))

    # --- 6. Total material ---------------------------------------------------
    # Layer count. Prefer Cura's own ;LAYER_COUNT comment; otherwise derive from
    # the Z range (distinct-Z counting is meaningless for continuous/spiral paths
    # where Z ramps every move). layer_height, when known, gives an exact figure.
    if layer_count_comment is not None:
        layer_est = layer_count_comment
    elif z_values:
        z_span = max(z_values) - min(z_values)
        if layer_height and layer_height > 0:
            layer_est = max(1, round(z_span / layer_height) + 1)
        else:
            levels = sorted(set(round(v, 2) for v in z_values))
            gaps = [b - a for a, b in zip(levels, levels[1:]) if b - a > 1e-3]
            step = min(gaps) if gaps else 0
            layer_est = (max(1, round(z_span / step) + 1) if step else len(levels))
    else:
        layer_est = 0

    clay_ml = total_extruded * filament_area / 1000.0
    if cartridge_ml:
        if clay_ml > cartridge_ml:
            issues.append(Issue(FAIL, "Material",
                f"Print needs ~{clay_ml:.0f} ml of clay but the cartridge holds "
                f"{cartridge_ml:.0f} ml. It will run out mid-print."))
        elif clay_ml > 0.85 * cartridge_ml:
            issues.append(Issue(WARN, "Material",
                f"Print needs ~{clay_ml:.0f} ml, close to the {cartridge_ml:.0f} ml "
                f"cartridge. Make sure it's full."))

    # Height is the top of the clay, not of the last move: Tronxy's end block
    # lifts 10 mm, so every Tronxy file read 10 mm taller than the pot
    # (Coil Bowl 72.4 for a 62.4 mm print). max_z keeps its key because the
    # app shows it as "Height"; max_z_moves is the old all-moves figure.
    height = None if ext_z_max == -math.inf else round(ext_z_max, 1)
    stats = {
        "min_x": None if min_xyz[0] is math.inf else round(min_xyz[0], 1),
        "max_x": None if max_xyz[0] == -math.inf else round(max_xyz[0], 1),
        "min_y": None if min_xyz[1] is math.inf else round(min_xyz[1], 1),
        "max_y": None if max_xyz[1] == -math.inf else round(max_xyz[1], 1),
        "max_z": height,
        "height": height,
        "max_z_moves": None if max_xyz[2] == -math.inf else round(max_xyz[2], 1),
        "first_z": None if first_ext is None else round(first_ext[0], 3),
        "min_extrude_z": None if ext_z_min == math.inf else round(ext_z_min, 3),
        "layers": layer_est,
        "median_e_per_mm": None if median_epm is None else round(median_epm, 3),
        "clay_ml": round(clay_ml, 1),
        "clay_g": round(clay_ml * 1.9, 1),  # ~1.9 g/ml wet stoneware
        "motion_hitches": {"z_jerk": z_hitch, "retrace": retrace, "steep": z_fast},
        "line_width_used": nozzle,
        "line_width_from_file": lw_from_file,
    }
    if not issues:
        issues.append(Issue(INFO, "OK", "No problems found. Ready to print."))
    return ValidationReport(issues, stats)
