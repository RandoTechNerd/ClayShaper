"""
STL -> clay G-code slicer.

Adapted from the standalone EazaoSlicer (Dev/EazaoSlicer/slicer_core.py) so it
lives inside ClayShaper and shares its printer profiles, start/end blocks and the
exact volumetric extrusion Eazao's own Cura uses. Two strategies, both clay-safe:

  * Vase mode  - each body layer is a single spiralized outer contour (one
                 continuous bead, no travels) sitting on a solid staggered base.
  * Wall mode  - each layer is its perimeter loops (for non-round vessels); short
                 travels between loops get a tiny retract to limit ooze.

Requires trimesh + shapely (already in requirements).
"""

import math
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import affine_transform

from clay_lib import PRINTER_PROFILES


class STLSlicer:
    def __init__(self, stl_path, profile, nozzle=3.0, layer_height=1.0,
                 line_width=None, first_layer_height=None, scale=1.0):
        self.profile = profile
        self.nozzle = nozzle
        self.layer_height = layer_height
        self.first_layer_height = first_layer_height or layer_height
        self.line_width = line_width if line_width is not None else nozzle
        self.filament_area = math.pi * (profile.get("filament_dia", 1.75) / 2.0) ** 2

        self.mesh = trimesh.load(stl_path, force="mesh")
        if scale and abs(scale - 1.0) > 1e-6:
            self.mesh.apply_scale(scale)

        # Center the model's XY footprint on the bed centre and drop it to Z=0.
        bmin, bmax = self.mesh.bounds
        bcenter = (bmin + bmax) / 2.0
        self.mesh.apply_translation([
            profile["center_x"] - bcenter[0],
            profile["center_y"] - bcenter[1],
            -bmin[2],
        ])

    # ------------------------------------------------------------------ slice
    def slice(self, bottom_layers=3, staggered=True, staggered_offset_factor=0.5,
              vase_mode=True, path_resolution=1.5, fold_softening=None):
        # Fold softening also widens the crevice-sealing radius: creases
        # narrower than 2x this are closed at the OUTLINE level, so deep folds
        # become shallow grooves instead of pits some layer must bridge over.
        lw = self.line_width
        if fold_softening is None:
            self._close_r = lw / 4.0
        elif fold_softening >= 1.4:      # Gentle
            self._close_r = lw / 2.0
        elif fold_softening >= 0.9:      # Medium
            self._close_r = lw
        else:                            # Strong
            self._close_r = 1.6 * lw
        z_max = self.mesh.bounds[1][2]
        layers = []

        # Solid base: concentric rings, offset half a line-width on alternating
        # layers so the seams don't stack (the "staggered" base). Parity is
        # anchored to the TOP base layer: the layer the wall lands on is never
        # the inset one, so the wall always has clay under it.
        for i in range(bottom_layers):
            z_target = self.first_layer_height + i * self.layer_height
            polys = self._section_polygons(z_target - 0.1)
            if not polys:
                continue
            paths = self._concentric_fill(polys, i, bottom_layers, staggered,
                                          staggered_offset_factor)
            paths = [self._resample(p, min(1.0, path_resolution)) for p in paths]
            # Inset ("staggered-in") layers sit half a line width inward — they
            # need a touch more clay to bond with the layers above/below, or the
            # base reads like an under-filled Oreo.
            inset = bool(staggered and (bottom_layers - 1 - i) % 2 == 1)
            layers.append({"z": z_target, "paths": paths, "type": "bottom",
                           "inset": inset})

        # Body. Perimeter centerlines are inset half a line width from the
        # section outline (like Cura), so the printed bead's outer face sits ON
        # the model surface — and exactly on the base's outermost ring, which
        # is inset by the same half line width.
        start_z = self.first_layer_height + (bottom_layers - 1) * self.layer_height \
            if bottom_layers > 0 else 0.0
        prev_start = None   # seam anchor: keeps direction + start aligned per layer
        prev_poly = None    # previous layer's chosen section, for sanity checks
        vase_rings = []     # collect, then smooth across layers before emitting
        for z_target in np.arange(start_z + self.layer_height, z_max, self.layer_height):
            polys = self._section_polygons(z_target - 0.1)
            if not polys and vase_mode and vase_rings:
                # Section failed entirely: repair by repeating the last ring at
                # this height rather than leaving a missing layer.
                vase_rings.append((float(z_target), vase_rings[-1][1]))
                continue
            if not polys:
                continue
            if vase_mode:
                largest = self._pick_section(polys, prev_poly, z_target)
                if largest is None:
                    # Degenerate fragment we couldn't recover: repeat last ring.
                    if vase_rings:
                        vase_rings.append((float(z_target), vase_rings[-1][1]))
                    continue
                prev_poly = largest
                ring = self._normalize_ring(self._perimeter_ring(largest), prev_start)
                if prev_start is None:
                    # First ring: park the seam on the FLATTEST stretch of the
                    # outline — a seam on a crease tears visually and physically.
                    ring = self._seam_to_flattest(ring)
                prev_start = ring.coords[0]
                vase_rings.append((float(z_target), ring))
            else:
                perims = []
                for p in polys:
                    inset = self._clean_inset(p)
                    if inset is None:
                        perims.append(self._resample(p.exterior, path_resolution))
                        continue
                    geoms = inset.geoms if isinstance(inset, MultiPolygon) else [inset]
                    for g in geoms:
                        perims.append(self._resample(self._normalize_ring(g.exterior), path_resolution))
                        for interior in g.interiors:
                            perims.append(self._resample(self._normalize_ring(interior), path_resolution))
                layers.append({"z": float(z_target), "paths": perims, "type": "wall"})

        # Smooth spiralized contours (what Cura does by default): blend each
        # vase ring with its vertical neighbors so per-layer offset decisions
        # can't flip-flop in and out of folds — that alternation left stepped
        # pockets of exposed coil ends on fold cheeks.
        if vase_mode and vase_rings:
            for z_target, ls in self._smooth_contours(vase_rings, path_resolution,
                                                      max_step=fold_softening):
                layers.append({"z": z_target, "paths": [ls], "type": "vase"})

        # Nothing came out at ANY height: the mesh could not be sectioned at
        # all. Fail loudly — an empty slice otherwise sails through validation
        # as a clean "PASS" with zero layers, which tells the user nothing.
        if not layers:
            err = getattr(self, "_last_section_error", None)
            detail = f" ({type(err).__name__}: {err})" if err else ""
            raise RuntimeError(
                "Could not slice this model — no cross-sections could be read "
                f"from the mesh{detail}. The STL may be corrupt or empty.")
        return layers

    def _smooth_contours(self, vase_rings, resolution, passes=2, max_step=None):
        """Vertical contour smoothing with nearest-point correspondence: each
        ring point is pulled toward the CLOSEST point on the rings above and
        below. Robust to seam drift and arc redistribution (index-matched
        blending averages unrelated points and shreds the wall). The first
        ring is anchored so the wall still lands exactly on the base."""
        # numpy stand-in for scipy's cKDTree: identical results on our point
        # counts, and it keeps ~30 MB of scipy out of the browser build.
        from nearest import NearestPoints as cKDTree

        n_pts = int(np.clip(max(r.length for _, r in vase_rings) / max(resolution, 0.3),
                            90, 720))

        def resample_n(ring):
            c = np.asarray(ring.coords)
            seg = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t = np.linspace(0.0, cum[-1], n_pts, endpoint=False)
            return np.column_stack([np.interp(t, cum, c[:, 0]),
                                    np.interp(t, cum, c[:, 1])])

        S = [resample_n(r) for _, r in vase_rings]   # list of (N,2)
        anchor = S[0].copy()
        L = len(S)
        for _ in range(passes):
            trees = [cKDTree(s) for s in S]
            new = []
            for i in range(L):
                cur = S[i]
                acc = 0.5 * cur
                wsum = 0.5
                for j in (i - 1, i + 1):
                    if 0 <= j < L:
                        _, idx = trees[j].query(cur)
                        acc = acc + 0.25 * S[j][idx]
                        wsum += 0.25
                new.append(acc / wsum)
            S = new
            S[0] = anchor       # keep the base-landing ring exactly where it was

        def clamp_to_below(max_d):
            """Bottom-up sweep: pull every point to within max_d of the (already
            clamped) ring below — Cura's 'make overhang printable' for coils."""
            for i in range(1, L):
                tree = cKDTree(S[i - 1])
                d, idx = tree.query(S[i])
                over = d > max_d
                if over.any():
                    base_pts = S[i - 1][idx[over]]
                    vec = S[i][over] - base_pts
                    scale = (max_d / d[over])[:, None]
                    S[i] = S[i].copy()
                    S[i][over] = base_pts + vec * scale

        def lone_pass(rounds=5):
            """Hunt LONE deviations — points >1.2mm from BOTH vertical neighbors
            on the same side (slit-makers) — pull toward the neighbor midpoint."""
            for _ in range(rounds):
                trees = [cKDTree(s) for s in S]
                changed = False
                for i in range(1, L - 1):
                    cur = S[i]
                    dp, ip = trees[i - 1].query(cur)
                    dn, ic = trees[i + 1].query(cur)
                    vp = cur - S[i - 1][ip]
                    vn = cur - S[i + 1][ic]
                    lone = (np.einsum("ij,ij->i", vp, vn) > 0) & (dp > 1.2) & (dn > 1.2)
                    if lone.any():
                        target = 0.5 * (S[i - 1][ip] + S[i + 1][ic])
                        S[i] = S[i].copy()
                        S[i][lone] = 0.35 * S[i][lone] + 0.65 * target[lone]
                        changed = True
                if not changed:
                    break

        lone_pass()
        if max_step is not None and max_step > 0:
            # Clamp LAST so the overhang guarantee actually holds: the lone
            # pass can push points back over a fold, so alternate and finish
            # with a clamp (bottom-up => every layer ends within max_step of
            # the final position of the layer below).
            for _ in range(2):
                clamp_to_below(max_step)
                lone_pass(rounds=2)
            clamp_to_below(max_step)

        out = []
        for i, (z, _) in enumerate(vase_rings):
            closed = np.vstack([S[i], S[i][:1]])
            out.append((z, LineString(closed)))
        return out

    def _clean_inset(self, poly):
        """Inset the outline by half a line width, then morphologically close
        it (dilate + erode by half a bead) to seal crevices narrower than the
        bead. Without this, deep folds make the offset boundary double back on
        itself — hairpin reversals the nozzle can't print (the factory Cura
        files contain zero of these)."""
        r = self.line_width / 2.0
        rc = getattr(self, "_close_r", self.line_width / 4.0)
        inset = poly.buffer(-r)
        if inset.is_empty:
            return None
        inset = inset.buffer(rc).buffer(-rc)      # closing: seal narrow folds
        if inset.is_empty:
            return None
        return inset.simplify(0.05)

    @staticmethod
    def _seam_to_flattest(ring, window=5):
        """Rotate a closed ring so it starts (and therefore seams) on the
        flattest stretch of the outline, measured as the smallest total turning
        angle over a sliding window of vertices."""
        coords = np.asarray(ring.coords)
        closed = np.allclose(coords[0], coords[-1])
        pts = coords[:-1] if closed else coords
        n = len(pts)
        if n < window + 2:
            return ring
        v = np.roll(pts, -1, axis=0) - pts
        L = np.linalg.norm(v, axis=1)
        L[L < 1e-9] = 1.0
        u = v / L[:, None]
        # turning angle at each vertex
        dots = np.clip(np.einsum("ij,ij->i", u, np.roll(u, -1, axis=0)), -1.0, 1.0)
        turn = np.arccos(dots)
        # total turning over a window, minimized = flattest stretch
        kern = np.ones(window)
        score = np.convolve(np.concatenate([turn, turn[:window]]), kern, mode="valid")[:n]
        k = int(np.argmin(score))
        pts = np.roll(pts, -k, axis=0)
        return LineString(np.vstack([pts, pts[:1]]))

    def _pick_section(self, polys, prev_poly, z_target):
        """Choose the section polygon for a vase layer, guarding against
        trimesh's silent polygonization failures (it can return only a small
        FRAGMENT of the real section with no error — following it teleports
        the wall sideways and tears a hole).

        A candidate is suspicious when its area collapses versus the previous
        layer while its centroid jumps: real tapers shrink in place. Suspicious
        layers are re-sectioned at nudged heights; None means unrecoverable
        (caller repeats the previous ring)."""
        largest = max(polys, key=lambda p: p.area)
        if prev_poly is None:
            return largest

        def suspicious(p):
            ratio = p.area / max(prev_poly.area, 1e-9)
            c0, c1 = prev_poly.centroid, p.centroid
            shift = math.hypot(c1.x - c0.x, c1.y - c0.y)
            drop_and_move = ratio < 0.6 and shift > 2.0 * self.line_width
            collapse = ratio < 0.25 and prev_poly.area > 200.0
            return drop_and_move or collapse

        if not suspicious(largest):
            return largest
        for dz in (0.15, -0.15, 0.3, -0.3, 0.45):
            alt = self._section_polygons(z_target - 0.1 + dz)
            if not alt:
                continue
            cand = max(alt, key=lambda p: p.area)
            if not suspicious(cand):
                return cand
        return None

    def _perimeter_ring(self, poly):
        """Wall centerline for vase mode: the outline inset by half a line
        width, kept as ONE unbroken ring.

        Where deep folds pinch the section, a full inset splits the polygon in
        two — and printing only the largest piece silently drops a whole lobe
        of the wall (found on the Paper Bag Vase: up to 18% of a layer gone,
        with the nozzle U-turning at the hole's edges). If the inset would
        drop a significant piece, relax it for that layer until the ring stays
        whole; the bead runs slightly wide of center there, which clay forgives.
        """
        r_full = self.line_width / 2.0
        r_close = getattr(self, "_close_r", r_full / 2.0)
        for factor in (1.0, 0.66, 0.33, 0.0):
            r = r_full * factor
            inset = poly if r <= 0 else poly.buffer(-r)
            if inset.is_empty:
                continue
            # seal sub-bead crevices (offset-boundary hairpins)
            inset = inset.buffer(r_close).buffer(-r_close).simplify(0.05)
            if inset.is_empty:
                continue
            if isinstance(inset, MultiPolygon):
                pieces = sorted(inset.geoms, key=lambda g: g.area, reverse=True)
                if factor > 0 and len(pieces) > 1 and pieces[1].area > 5.0:
                    continue   # would drop a real lobe -> retry with less inset
                inset = pieces[0]
            return inset.exterior
        return poly.exterior

    @staticmethod
    def _normalize_ring(ring, prev_start=None):
        """Make every loop print the same way around (CCW) and start near the
        previous layer's start point. trimesh sections come back in arbitrary
        winding order — without this, alternating layers reverse direction and
        the nozzle does a U-turn at every seam (the factory files never do)."""
        coords = np.asarray(ring.coords)
        if len(coords) < 4:
            return ring
        closed = np.allclose(coords[0], coords[-1])
        pts = coords[:-1] if closed else coords
        # enforce CCW via the shoelace signed area
        area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                       - np.roll(pts[:, 0], -1) * pts[:, 1])
        if area2 < 0:
            pts = pts[::-1]
        # rotate so the seam stays put layer to layer
        if prev_start is not None:
            k = int(np.argmin(np.hypot(pts[:, 0] - prev_start[0],
                                       pts[:, 1] - prev_start[1])))
            pts = np.roll(pts, -k, axis=0)
        pts = np.vstack([pts, pts[:1]])
        return LineString(pts)

    @staticmethod
    def _rings_to_polygons(rings):
        """Assemble closed 2D rings into polygons with holes, using shapely.

        trimesh's own `polygons_full` needs the optional `rtree` package for its
        containment queries, which isn't available in WebAssembly — without it
        EVERY section that has more than one ring (any model with islands or
        holes, e.g. a Benchy's hull + cabin) raised, and the slice silently
        stopped a third of the way up. Ring counts per layer are tiny, so a
        direct O(n^2) containment test is both simpler and dependency-free.
        """
        polys = []
        for pts in rings:
            if len(pts) < 4:
                continue
            p = Polygon(pts)
            if not p.is_valid:
                p = p.buffer(0)
            if isinstance(p, MultiPolygon):
                p = max(p.geoms, key=lambda g: g.area) if p.geoms else None
            if p is None or p.is_empty or p.area <= 1e-9:
                continue
            polys.append(p)
        if not polys:
            return []

        # Nesting depth: even = solid outline, odd = hole in the ring above it.
        # Containment must be FULL and strictly area-ordered. Testing only a
        # representative point makes two partially-overlapping rings each look
        # "inside" the other (real case: a Benchy's cabin at ~35 mm), so both
        # scored odd and were discarded as holes — dropping the layer entirely.
        n = len(polys)
        depth = [0] * n
        contains = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and polys[j].area > polys[i].area and polys[j].contains(polys[i]):
                    contains[j][i] = True
                    depth[i] += 1

        out = []
        for i in range(n):
            if depth[i] % 2:
                continue                      # this ring is a hole
            holes = [polys[j].exterior.coords for j in range(n)
                     if depth[j] == depth[i] + 1 and contains[i][j]]
            poly = Polygon(polys[i].exterior.coords, holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly is not None and not poly.is_empty and poly.area > 1e-9:
                out.append(poly)
        return out

    def _section_polygons(self, z):
        # Some meshes have degenerate geometry at specific heights that trimesh
        # can't polygonize ("unable to recover polygon") — nudge and retry.
        # Per-height failures are normal and silently skipped, but we remember
        # the last error: if EVERY height fails (e.g. a missing trimesh
        # dependency) the caller must raise instead of returning an empty
        # slice, which used to surface as a mystifying "0 layers / PASS".
        for dz in (0.0, 0.03, -0.03, 0.08, -0.08):
            try:
                section = self.mesh.section(plane_origin=[0, 0, z + dz],
                                            plane_normal=[0, 0, 1])
                if section is None:
                    continue
                path2d, transform = section.to_2D()
                a, b, xoff = transform[0, 0], transform[0, 1], transform[0, 3]
                d, e, yoff = transform[1, 0], transform[1, 1], transform[1, 3]
                # Prefer trimesh's own polygon assembly (the behaviour every
                # sample model is tuned against), but fall back to our shapely
                # builder when it can't run — notably in WebAssembly, where the
                # optional `rtree` package is missing and ANY section with more
                # than one ring (islands/holes, e.g. a Benchy) would otherwise
                # raise and silently truncate the model.
                try:
                    polys = list(path2d.polygons_full)
                except Exception as exc:
                    self._last_section_error = exc
                    self.used_ring_fallback = True
                    polys = self._rings_to_polygons(path2d.discrete)
                if not polys:
                    continue
                return [affine_transform(p, [a, b, d, e, xoff, yoff])
                        for p in polys]
            except Exception as exc:
                self._last_section_error = exc
                continue
        self.section_failures = getattr(self, "section_failures", 0) + 1
        return []

    def _concentric_fill(self, polygons, layer_index, bottom_layers, staggered,
                         offset_factor):
        paths = []
        step = self.line_width
        # Count parity from the TOP: the last base layer (index bottom_layers-1)
        # must never be inset, because the wall lands on its outermost ring.
        from_top = bottom_layers - 1 - layer_index
        initial = offset_factor * self.line_width if (staggered and from_top % 2 == 1) else 0.0
        for poly in polygons:
            d = initial
            while True:
                buffered = poly.buffer(-d - self.line_width / 2)
                if buffered.is_empty or buffered.area < self.line_width ** 2:
                    break
                geoms = buffered.geoms if isinstance(buffered, MultiPolygon) else [buffered]
                for g in geoms:
                    paths.append(self._normalize_ring(g.exterior))
                    paths.extend(self._normalize_ring(i) for i in g.interiors)
                d += step
        return paths

    def _resample(self, ring, resolution):
        """Simplify then subdivide so no segment is longer than `resolution`
        (keeps clay pressure even)."""
        if ring is None:
            return None
        simplified = ring.simplify(max(0.01, resolution * 0.1), preserve_topology=True)
        coords = np.asarray(simplified.coords)
        if len(coords) < 2:
            return simplified
        out = [coords[0]]
        for i in range(1, len(coords)):
            p1, p2 = out[-1], coords[i]
            vec = p2 - p1
            dist = float(np.linalg.norm(vec))
            if dist > resolution:
                n = int(dist / resolution)
                for j in range(1, n + 1):
                    out.append(p1 + vec * (j / (n + 1)))
            else:
                out.append(p2)
        if np.allclose(coords[0], coords[-1]) and not np.allclose(out[0], out[-1]):
            out.append(out[0])
        return LineString(out)

    # ------------------------------------------------------------------ gcode
    def to_gcode(self, layers, first_layer_flow=1.0, source=None, continuous=True,
                 stagger_fill=1.0):
        """
        continuous: when True (vase mode), consecutive wall layers are JOINED
        with an extruding move instead of a travel whenever the seam jump is
        small — the whole vessel becomes one unbroken bead, so the nozzle never
        stops extruding and can never loop/drag at the seam.
        stagger_fill: extrusion multiplier for the inset (staggered-in) base
        layers, so they lay down enough clay to bond (fills the "Oreo gap").
        """
        profile = self.profile
        f_print = profile.get("print_speed", 1500)
        f_z = profile.get("z_speed", 300)
        e_per_mm = (self.layer_height * self.line_width) / self.filament_area

        # Toolpath bounds for the header.
        xs, ys, zs = [], [], []
        for layer in layers:
            for path in layer["paths"]:
                if path is None:
                    continue
                for cx, cy in path.coords:
                    xs.append(cx); ys.append(cy)
            zs.append(layer["z"])

        g = [";FLAVOR:Marlin", ";Generated by ClayShaper (STL slice)"]
        if source:
            g.append(f";SOURCE: {source}")
        if first_layer_flow != 1.0:
            g.append(f";First layer flow: {first_layer_flow*100:.0f}%")
        g += [f";Layer height: {self.layer_height:g}", f";Line width: {self.line_width:g}"]
        if xs:
            g += [f";MINX:{min(xs):.2f}", f";MINY:{min(ys):.2f}", f";MINZ:{min(zs):.2f}",
                  f";MAXX:{max(xs):.2f}", f";MAXY:{max(ys):.2f}", f";MAXZ:{max(zs):.2f}"]
        g.append(f";LAYER_COUNT:{len(layers)}")
        g.append(profile["start_gcode"])
        g.append("M107")

        total_e = 0.0
        first = True
        last_xy = None    # nozzle position after the previous path (for joins)
        for li, layer in enumerate(layers):
            g.append(f";LAYER:{li}")
            z = layer["z"]
            is_vase = layer["type"] == "vase"
            h = self.first_layer_height if li == 0 else self.layer_height
            layer_e = (h * self.line_width / self.filament_area) \
                * (first_layer_flow if li == 0 else 1.0)
            if layer.get("inset"):
                layer_e *= stagger_fill
            g.append(";TYPE:SKIN" if layer["type"] == "bottom" else ";TYPE:WALL-OUTER")

            for pi, path in enumerate(layer["paths"]):
                if path is None:
                    continue
                coords = list(path.coords)
                if len(coords) < 2:
                    continue
                sx, sy = coords[0]
                join_d = (math.hypot(sx - last_xy[0], sy - last_xy[1])
                          if last_xy is not None else None)
                if first:
                    g.append(f"G0 F{f_print} X{sx:.3f} Y{sy:.3f} Z{z:.3f}")
                    g.append(f"G1 F{f_z} Z{z:.3f}")
                    g.append(f"G1 F{f_print} E0")
                    first = False
                elif (continuous and is_vase and pi == 0
                        and join_d is not None and join_d < 2.0):
                    # Continuous spiral: extrude across the tiny seam jump —
                    # the bead never breaks between layers.
                    total_e += join_d * layer_e
                    g.append(f"G1 X{sx:.3f} Y{sy:.3f} Z{z:.3f} E{total_e:.5f}")
                else:
                    # Travel with a short retract to limit ooze (clay can't
                    # do big retracts).
                    total_e -= 0.1
                    g.append(f"G1 F{f_print} E{total_e:.5f}")
                    g.append(f"G0 F{f_print} X{sx:.3f} Y{sy:.3f} Z{z:.3f}")
                    total_e += 0.1
                    g.append(f"G1 F{f_print} E{total_e:.5f}")

                n = len(coords) - 1
                for j in range(n):
                    x1, y1 = coords[j]
                    x2, y2 = coords[j + 1]
                    total_e += math.hypot(x2 - x1, y2 - y1) * layer_e
                    if is_vase:
                        zj = z + (j + 1) / n * self.layer_height
                        g.append(f"G1 X{x2:.3f} Y{y2:.3f} Z{zj:.3f} E{total_e:.5f}")
                    else:
                        g.append(f"G1 X{x2:.3f} Y{y2:.3f} E{total_e:.5f}")
                last_xy = coords[-1]

        g.append(profile["end_gcode"])
        return "\n".join(g)


def slice_stl(stl_path, profile, nozzle=3.0, layer_height=1.0, bottom_layers=3,
              staggered=True, staggered_offset_factor=0.5, vase_mode=True,
              path_resolution=1.5, line_width=None, first_layer_flow=1.0,
              source=None, first_layer_height=None, continuous=True,
              fold_softening=None, scale=1.0, stagger_fill=1.0, diagnostics=None):
    """Convenience wrapper: returns (gcode_str, layers) for preview + export.

    diagnostics: optional dict, filled with how many heights failed to section
    and why — so the UI can tell the user when a model only partly sliced
    instead of silently handing back a stump.
    """
    slicer = STLSlicer(stl_path, profile, nozzle=nozzle, layer_height=layer_height,
                       line_width=line_width, first_layer_height=first_layer_height,
                       scale=scale)
    layers = slicer.slice(bottom_layers, staggered, staggered_offset_factor,
                          vase_mode, path_resolution, fold_softening=fold_softening)
    if diagnostics is not None:
        z_max = slicer.mesh.bounds[1][2]
        diagnostics["failed_heights"] = getattr(slicer, "section_failures", 0)
        diagnostics["expected_heights"] = max(1, int(z_max / max(layer_height, 1e-6)))
        diagnostics["layers"] = len(layers)
        err = getattr(slicer, "_last_section_error", None)
        diagnostics["last_error"] = f"{type(err).__name__}: {err}" if err else None
        diagnostics["model_top_mm"] = float(z_max)
        diagnostics["sliced_top_mm"] = float(max((l["z"] for l in layers), default=0.0))
    return slicer.to_gcode(layers, first_layer_flow=first_layer_flow, source=source,
                           continuous=continuous, stagger_fill=stagger_fill), layers
