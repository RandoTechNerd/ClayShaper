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

_TOKEN = re.compile(r"([A-Za-z])(-?\d+\.?\d*)")


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
    e_abs = True
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
            e_abs = True; continue
        if verb == "M83":
            e_abs = False; continue
        if verb == "G92":
            if "X" in w: x = w["X"]
            if "Y" in w: y = w["Y"]
            if "Z" in w: z = w["Z"]
            if "E" in w: e = w["E"]
            continue
        if verb not in ("G0", "G1"):
            continue

        nx, ny, nz = x, y, z
        if "X" in w: nx = w["X"] if absolute else x + w["X"]
        if "Y" in w: ny = w["Y"] if absolute else y + w["Y"]
        if "Z" in w: nz = w["Z"] if absolute else z + w["Z"]

        de = 0.0
        if "E" in w:
            de = (w["E"] - e) if e_abs else w["E"]
            e = w["E"] if e_abs else e + w["E"]

        if de > 1e-9 and (nx != x or ny != y or nz != z):
            if not pts or pts[-1] is None:
                pts.append((x, y, z, cur_kind))   # run start: include the origin
            pts.append((nx, ny, nz, cur_kind))
        elif pts and pts[-1] is not None:
            pts.append(None)                       # travel: break the line

        x, y, z = nx, ny, nz

    # Downsample long paths (keep run boundaries so gaps stay put).
    n_real = sum(1 for p in pts if p is not None)
    step = max(1, n_real // max_points)
    out = {"base": ([], [], []), "wall": ([], [], [])}
    i = 0
    prev_kind = None
    for p in pts:
        if p is None:
            if prev_kind is not None:
                arr = out[prev_kind]
                arr[0].append(None); arr[1].append(None); arr[2].append(None)
            prev_kind = None
            continue
        i += 1
        px, py, pz, kind = p
        if step > 1 and (i % step) and prev_kind == kind:
            continue
        if prev_kind is not None and kind != prev_kind:
            arr = out[prev_kind]
            arr[0].append(None); arr[1].append(None); arr[2].append(None)
        arr = out[kind]
        arr[0].append(px); arr[1].append(py); arr[2].append(pz)
        prev_kind = kind
    return out


def _analyze_geometry(segments, bed_x, bed_y, line_width=3.0):
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
    if len(skin_ids) >= 2:
        maxr = {}
        for li in skin_ids:
            pts_all = []
            for x0, y0, x1, y1, skin, *_ in layers[li]:
                if skin:
                    pts_all.append((x0, y0)); pts_all.append((x1, y1))
            p = np.asarray(pts_all)
            c = p.mean(axis=0)
            maxr[li] = float(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]).max())
        # "Stacked" = ring shift much smaller than a stagger step (half a line
        # width); natural flare of a sloped wall is ~0.3-0.5mm and still counts.
        stack_tol = 0.35 * line_width
        pairs = list(zip(skin_ids, skin_ids[1:]))
        stacked = sum(1 for a, b in pairs if abs(maxr[a] - maxr[b]) < stack_tol)
        if pairs and stacked == len(pairs):
            issues.append(Issue(SUGGEST, "Base seams",
                "Base rings stack directly on each other (no stagger). Offsetting "
                "alternate base layers half a line width spreads the seams and makes "
                "a stronger, more watertight bottom — enable “Staggered base” when slicing."))
    return issues


def validate_gcode(text, profile, nozzle=None, layer_height=None,
                   max_spike_examples=5):
    """
    Validate a G-code string against a printer profile.

    nozzle / layer_height are optional; when given, the flow check compares the
    file's real extrusion against the volume it *should* be laying down.
    """
    issues = []
    lines = text.splitlines()

    bed_x = profile["bed_x"]
    bed_y = profile["bed_y"]
    max_z = profile["max_z"]
    max_feed = profile.get("max_feedrate", 3600)
    max_print = profile.get("max_print_speed", 2400)
    min_print = profile.get("min_print_speed", 0)
    filament_area = math.pi * (profile.get("filament_dia", 1.75) / 2.0) ** 2
    cartridge_ml = profile.get("cartridge_ml")

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
    if not have["M163"] or not have["M164"]:
        issues.append(Issue(WARN, "Clay safety",
            "Missing M163/M164 mixing-ratio commands. Eazao's dual-material hotend "
            "expects them; extrusion may be wrong or blocked."))

    # --- 2. Walk the toolpath -----------------------------------------------
    absolute = True          # G90 default
    x = y = z = 0.0
    e = 0.0
    e_abs = True             # M82 default (absolute E)
    feed = 0.0
    have_pos = False

    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]
    oob_count = 0
    oob_example = None
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
            e_abs = True; continue
        if verb == "M83":
            e_abs = False; continue
        if verb == "G92":
            # Reset logical position (does not move). Update tracked coords.
            if "X" in w: x = w["X"]
            if "Y" in w: y = w["Y"]
            if "Z" in w: z = w["Z"]
            if "E" in w: e = w["E"]
            continue
        if verb not in ("G0", "G1"):
            continue

        if "F" in w:
            feed = w["F"]

        nx, ny, nz = x, y, z
        if "X" in w: nx = w["X"] if absolute else x + w["X"]
        if "Y" in w: ny = w["Y"] if absolute else y + w["Y"]
        if "Z" in w: nz = w["Z"] if absolute else z + w["Z"]

        moved = (nx != x) or (ny != y) or (nz != z)
        dist_xy = math.hypot(nx - x, ny - y)

        # Extrusion delta
        de = 0.0
        if "E" in w:
            if e_abs:
                de = w["E"] - e
                e = w["E"]
            else:
                de = w["E"]
                e += w["E"]
        is_extrude = de > 1e-9 and dist_xy > 1e-6

        # Bounds (only meaningful once we have a real position; skip pure Z
        # homing dance which uses machine coordinates before first XY move).
        if "X" in w or "Y" in w:
            have_pos = True
        if have_pos and moved:
            for axis, val, lo, hi in (
                (0, nx, 0.0, bed_x), (1, ny, 0.0, bed_y), (2, nz, 0.0, max_z)
            ):
                min_xyz[axis] = min(min_xyz[axis], val)
                max_xyz[axis] = max(max_xyz[axis], val)
            # small tolerance for float noise / seam
            tol = 0.5
            if (nx < -tol or nx > bed_x + tol or ny < -tol or ny > bed_y + tol
                    or nz < -tol or nz > max_z + tol):
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

        x, y, z = nx, ny, nz

    # --- 3. Bounds verdict ---------------------------------------------------
    if oob_count:
        li, ox, oy, oz = oob_example
        issues.append(Issue(FAIL, "Bounds",
            f"{oob_count} move(s) leave the {bed_x:g}x{bed_y:g}x{max_z:g} build "
            f"volume, e.g. X{ox} Y{oy} Z{oz}. Re-center or scale the model.", li))

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
                    if p90 > expected * 1.5:
                        issues.append(Issue(WARN, "Flow",
                            f"10% of the path extrudes at {p90:.2f} E/mm — over 1.5x "
                            f"the expected {expected:.2f}. Sustained over-extrusion "
                            f"blobs and drags in clay; check first-layer/flow settings."))

    # --- 5b. Geometry: support ("thin air"), solid areas, stagger -------------
    try:
        lw = nozzle if nozzle else 3.0
        issues.extend(_analyze_geometry(segments, bed_x, bed_y, line_width=lw))
    except Exception:
        pass   # geometry analysis is best-effort; never block validation on it

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

    stats = {
        "min_x": None if min_xyz[0] is math.inf else round(min_xyz[0], 1),
        "max_x": None if max_xyz[0] == -math.inf else round(max_xyz[0], 1),
        "min_y": None if min_xyz[1] is math.inf else round(min_xyz[1], 1),
        "max_y": None if max_xyz[1] == -math.inf else round(max_xyz[1], 1),
        "max_z": None if max_xyz[2] == -math.inf else round(max_xyz[2], 1),
        "layers": layer_est,
        "median_e_per_mm": None if median_epm is None else round(median_epm, 3),
        "clay_ml": round(clay_ml, 1),
        "clay_g": round(clay_ml * 1.9, 1),  # ~1.9 g/ml wet stoneware
    }
    if not issues:
        issues.append(Issue(INFO, "OK", "No problems found. Ready to print."))
    return ValidationReport(issues, stats)
