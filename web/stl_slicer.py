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
import offset2d


def _points_in_ring(pts, ring):
    """Even-odd point-in-polygon for an (N, 2) array of points against one
    closed ring of (M, 2) vertices, in numpy alone. Used where a GEOS call
    would be the only one on the path: GEOS errors abort the interpreter
    under WebAssembly, and no try/except can catch them there."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = ring[:-1, 0][None, :], ring[:-1, 1][None, :]
    x1, y1 = ring[1:, 0][None, :], ring[1:, 1][None, :]
    straddles = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.count_nonzero(straddles & (x < x_cross), axis=1) % 2) == 1


class STLSlicer:
    def __init__(self, stl_path, profile, nozzle=3.0, layer_height=1.0,
                 line_width=None, first_layer_height=None, scale=1.0,
                 base_layer_height=None):
        self.profile = profile
        self.nozzle = nozzle
        self.layer_height = layer_height
        # The base and the walls want opposite things. Walls want thin
        # layers, because a thin layer steps outward less and so keeps more
        # of each bead sitting on the one below. The base wants thick ones:
        # a bead is line_width across whatever its height, so at 0.6 mm from
        # a 3 mm nozzle it is five times wider than it is tall, and the
        # nozzle face smears it into a sliver instead of laying a round
        # cord. Keeping the two apart lets the walls go fine without
        # shaving the base down with them.
        self.base_layer_height = base_layer_height or layer_height
        self.first_layer_height = first_layer_height or self.base_layer_height
        self.line_width = line_width if line_width is not None else nozzle
        self.filament_area = math.pi * (profile.get("filament_dia", 1.75) / 2.0) ** 2

        self.mesh = trimesh.load(stl_path, force="mesh")
        # trimesh does not raise on a bad file. A binary STL whose header
        # count disagrees with its size, a truncated download, an empty file
        # or a renamed PNG all load as a mesh with ZERO faces, and the bounds
        # read below then died with "cannot unpack non-iterable NoneType",
        # which tells the user nothing. Say what is actually wrong instead.
        if len(getattr(self.mesh, "faces", ())) == 0:
            raise ValueError("This file has no triangles: it is empty, "
                             "truncated or not an STL")
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

        # Does it fit the printer? Until now nothing compared the model with
        # the build volume before slicing: a 320 mm cup on the 165x165x280
        # Eazao sliced for 17-50 s (over a gigabyte of browser heap) only for
        # the validator to FAIL every move at the end. Pure numpy, so it is
        # free and safe under WebAssembly. XY gets one line width of slack
        # because the bead is that wide, half of it each side. fit_scale is
        # the factor the CURRENT size can be multiplied by and still fit, so
        # the UI can say "try Scale <= N%"; above 1 means there is room left.
        self.extents = np.asarray(self.mesh.extents, dtype=float)
        self.fits_bed = None
        self.fit_scale = None
        try:
            room = np.array([float(profile["bed_x"]) - self.line_width,
                             float(profile["bed_y"]) - self.line_width,
                             float(profile["max_z"])])
        except (KeyError, TypeError, ValueError):
            room = None     # a profile with no build volume: nothing to check
        if room is not None:
            self.fits_bed = bool(np.all(self.extents <= room))
            ratios = np.where(self.extents > 0,
                              room / np.maximum(self.extents, 1e-12), np.inf)
            self.fit_scale = float(np.min(ratios))
        # Filled by slice(): the Z the print really starts at when that is
        # above the first layer (the model does not rest on the bed).
        self.floating_start = None

    # ------------------------------------------------------------------ slice
    def slice(self, bottom_layers=3, staggered=True, staggered_offset_factor=0.5,
              vase_mode=True, path_resolution=1.5, fold_softening=None,
              measure_support=True, spiral=True):
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
            z_target = self.first_layer_height + i * self.base_layer_height
            # Sample the MIDDLE of the slab this layer represents, the way a
            # slicer should. A fixed 0.1 mm below the top meant the first layer
            # took its outline from 90% of the way up its own slab, so any
            # model with a curved or tapered bottom (a boat hull, a rounded
            # foot) printed a flat pad much wider than the real base and looked
            # squashed. It also drifted with layer height: 75% up a 0.4 mm
            # layer, 96% up a 2.7 mm one.
            slab = self.first_layer_height if i == 0 else self.base_layer_height
            polys = self._section_polygons(max(z_target - slab / 2.0, 1e-3))
            if not polys:
                continue
            paths = self._concentric_fill(polys, i, bottom_layers, staggered,
                                          staggered_offset_factor)
            paths = [self._resample(p, min(1.0, path_resolution)) for p in paths]
            paths = self._chain_base_rings(paths, reach=self.STEP_OVER_BEADS * self.line_width,
                                           trim=0.5 * self.line_width)
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
        start_z = self.first_layer_height + (bottom_layers - 1) * self.base_layer_height \
            if bottom_layers > 0 else 0.0
        prev_start = None   # seam anchor: keeps direction + start aligned per layer
        prev_poly = None    # previous layer's chosen section, for sanity checks
        vase_rings = []     # collect, then smooth across layers before emitting
        for z_target in np.arange(start_z + self.layer_height, z_max, self.layer_height):
            polys = self._section_polygons(z_target - self.layer_height / 2.0)
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
            rings = list(self._smooth_contours(vase_rings, path_resolution,
                                              max_step=fold_softening))
            if spiral:
                rings = self._spiralize(rings, min_seg=min(0.3, 0.6 * path_resolution))
            for z_target, ls in rings:
                layers.append({"z": z_target, "paths": [ls], "type": "vase"})

        # --- Overhang safety -------------------------------------------------
        # Clay cannot bridge air. For each layer, work out how much of the bead
        # it actually lays down lands on clay from the layer below. A bead is
        # line_width across, so a layer that steps outward by half a bead only
        # gets half of itself supported, and the rim droops. A drooping rim is
        # exactly what makes a base or wall look detached.
        # Every overlay here runs in Clipper on a fixed 1 um integer grid, not
        # in GEOS. At full float precision a GEOS union of buffered beads can
        # hit "found non-noded intersection" on perfectly ordinary toolpaths.
        # On desktop that surfaces as a catchable Python error, but under
        # WebAssembly it aborts the interpreter outright, and no try/except
        # can catch it: the whole app freezes with the model half-loaded.
        # Integer arithmetic has no such failure path. 1 um is far finer than
        # any clay bead, so it costs nothing.
        self.min_support_frac = 1.0
        self.min_support_z = None
        if measure_support:
            self._measure_support(layers)

        # Drop layers that came out with nothing to print. A base section
        # narrower than one bead (Eazao's Pleated Vase stands on a ring foot
        # about 1.5 mm wide, under a 3 mm nozzle) gives _concentric_fill no
        # rings at all, and those empty layers used to go out as layer 0..2.
        # to_gcode gives layer 0 the first-layer height and flow, so that
        # treatment landed on nothing and the first real bead went down as
        # an ordinary layer. Same "drawable" test as to_gcode uses.
        def _has_bead(lay):
            return any(p is not None and len(p.coords) >= 2 for p in lay["paths"])
        kept = [lay for lay in layers if _has_bead(lay)]
        self.empty_layers_dropped = 0
        # Unless EVERY layer is empty. An 80 mm cup exported in inches is a
        # 3.15 mm cube: it sections fine, but nothing in it is as wide as one
        # bead, so all 3 layers come out empty. Dropping them all would turn
        # that into the "could not read any cross-sections, corrupt file"
        # error below, which is wrong (the file is fine, the units are not)
        # and would stop the validator from saying "this prints nothing". So
        # that case keeps its old output and is left to the validator and the
        # app's units hint.
        if kept:
            self.empty_layers_dropped = len(layers) - len(kept)
            layers = kept

        # Floating start: the first bead is laid above the first layer, so it
        # drops through air onto the bed and the piece has no floor. The
        # Pleated Vase at defaults first extrudes at Z3.6 and nothing flagged
        # it (even the support measure has no layer below the first one to
        # compare with). Detection only for now; for a vase layer the ramp
        # starts at its z, which is where to_gcode puts the nozzle down.
        self.floating_start = None
        if layers:
            z0 = float(min(lay["z"] for lay in layers))
            if z0 > self.first_layer_height + 0.01:
                self.floating_start = z0

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

    @staticmethod
    def _spiralize(rings, min_seg=0.3):
        """Turn stacked rings into one true helix.

        Vase mode already ramps Z across each ring, but every ring holds its
        own layer's shape for the whole turn and then swaps to the next one
        where it wraps. That swap is a step in the wall, and because the seam
        is anchored to the same angle on every layer, all of those steps line
        up into a single vertical scar.

        Blending fixes it at the source: point j of a layer is walked from
        this ring toward the next one in step with how far round the turn it
        is, so the radius changes smoothly the whole way instead of all at
        once. The end of a layer then lands exactly on the start of the next
        (t reaches 1 at the closing point), leaving nothing to seam.

        The turn that lands on the base is left alone. Walking it outward
        straight away puts noticeably more of that first bead out past the
        base rim, and that joint is the one that gives clay trouble in the
        first place. One small step at the foot, sitting on solid base, beats
        one up the whole wall. The top turn has no ring above it to walk
        toward, so it is left alone too.

        min_seg is the shortest move the blend may leave behind (see the
        clean-up step below). slice() passes min(0.3, 0.6 x path resolution)
        so a user who asked for a fine 0.3 mm path is not thinned out.
        """
        from nearest import NearestPoints as cKDTree

        # Re-anchor every ring's start before blending. _smooth_contours
        # resamples and averages each ring, which drifts index 0 away from
        # the seam _normalize_ring picked: 0.7-2 mm on Coil Bowl and the
        # Eazao Spiral Vase. The blend below lands the end of each turn on
        # the next ring's index 0, so a drifted start made the nozzle run
        # past it and come straight back, two 180 deg reversals inside
        # 1.5 mm with the clay laid three times (Coil Bowl: 7 seams, spikes
        # 0.74-1.06 mm; Spiral Vase: 104 reversals). Rolling each ring so it
        # starts at the vertex nearest the previous ring's start keeps the
        # seam where it was chosen and removes the out-and-back. Ring 0 is
        # not touched, so the seam _seam_to_flattest placed still leads.
        #
        # The nearest vertex must also be heading the same way as the start
        # it continues. A ring can come out of _smooth_contours doubled back
        # on itself: Paper Bag Vase rings 211 and 215 run once round CCW, turn
        # at a slit and come back round CW 0.3 mm further in (716 mm long
        # against 362 mm for their neighbours, net area 95 mm2 against
        # 8100). There the nearest vertex (0.18 mm) sat on the backward lap,
        # so the turn below arrived going one way and the next one left going
        # the other: a 159 and a 175 deg reversal at the two seams that the
        # spiral-OFF print does not have. If the nearest vertex points back
        # against the previous start's direction, take the nearest one that
        # does not (0.60 mm there). On an ordinary ring the nearest vertex
        # always agrees, so nothing else moves.
        #
        # Ring 1 gets one more look. Turn 0 is not blended, so it closes on
        # its own start and then hops to ring 1's start, and the nearest
        # vertex is usually off to the side of that point rather than ahead
        # of it: the hop made two corners of about 90 deg (Coil Bowl 88/93,
        # Spiral Vase 89/94, Kuksa 84/96) and on the Draped Vase it went 0.2
        # mm backwards, 150/154 deg, an out-and-back on the first seam. On
        # Klipper that corner was the last wall slowdown left (1-2 per
        # print, 0 with this). Starting ring 1 up to three vertices further
        # on, whichever gives the gentlest pair of corners, turns the hop
        # into a straight run forward (worst corner 34 deg on those models).
        # Every later ring follows ring 1's start as above. The hop is kept
        # under 1.8 mm because to_gcode only joins turns closer than 2 mm
        # with clay; anything longer becomes a retract and travel.
        def corner(u, v):
            nu, nv = np.hypot(*u), np.hypot(*v)
            if nu < 1e-9 or nv < 1e-9:
                return 0.0
            return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))

        R = [np.asarray(ls.coords)[:-1].copy() for _, ls in rings]
        for i in range(1, len(R)):
            d = np.hypot(*(R[i] - R[i - 1][0]).T)
            k = int(np.argmin(d))
            heading = np.roll(R[i], -1, axis=0) - np.roll(R[i], 1, axis=0)
            agree = heading @ (R[i - 1][1] - R[i - 1][-1]) > 0
            if not agree[k] and agree.any():
                k = int(np.argmin(np.where(agree, d, np.inf)))
            if i == 1:
                p0, n = R[0][0], len(R[1])
                arrive = R[0][0] - R[0][-1]
                best = None
                for s in range(4):
                    c = (k + s) % n
                    hop, leave = R[1][c] - p0, R[1][(c + 1) % n] - R[1][c]
                    if s and np.hypot(*hop) > 1.8:
                        continue
                    if np.hypot(*hop) < 1e-9:
                        worst = corner(arrive, leave)
                    else:
                        worst = max(corner(arrive, hop), corner(hop, leave))
                    if best is None or worst < best[0]:
                        best = (worst, c)
                k = best[1]
            R[i] = np.roll(R[i], -k, axis=0)

        out = []
        for i, (z, _) in enumerate(rings):
            pts = np.vstack([R[i], R[i][:1]])
            if i > 0 and i + 1 < len(rings) and len(pts) > 2:
                nxt = np.vstack([R[i + 1], R[i + 1][:1]])
                # Walk each point toward the CLOSEST point on the next ring,
                # not the one with the same index. Matching by index assumes
                # both rings were cut at the same place and run at the same
                # pace, and on a crumpled surface they do not: on the Paper
                # Bag Vase the same-index point is 4.5 mm away while the real
                # nearest one is 0.17 mm, so an index blend drags the wall
                # sideways instead of easing it outward. This is the same
                # correspondence _smooth_contours uses, for the same reason.
                # (Projecting onto the next ring's polyline instead of its
                # vertices was tried too: it broke the Paper Bag Vase, support
                # 0.10 and an 11 mm face error, so vertices it stays.)
                _, idx = cKDTree(nxt[:-1]).query(pts)
                idx = np.asarray(idx)
                idx[-1] = 0          # land exactly where the next turn starts
                t = np.linspace(0.0, 1.0, len(pts))[:, None]
                pts = pts * (1.0 - t) + nxt[idx] * t

                # Snapping to vertices lets several neighbouring points pick
                # the same target, and near the end of the turn (t close to 1)
                # they then land almost on top of each other. Those leftovers
                # were 0.001-0.3 mm moves, 95-99% of them in the last 5% of
                # the turn, each carrying a full share of the turn's Z rise:
                # Z/XY up to 4000 against 0.007 in Eazao's own Cura files, and
                # a Z-jerk brake at each one (Twist Pot 224, Draped Pot 711,
                # Paper Bag 1057 per print). Drop any point closer than
                # min_seg to the last one kept, and the second-to-last if it
                # sits within min_seg of the end. The closing point is always
                # kept exactly, so the turn still ends on the next turn's
                # first point and the seam step stays 0.
                keep = [0]
                for j in range(1, len(pts) - 1):
                    if np.hypot(*(pts[j] - pts[keep[-1]])) >= min_seg:
                        keep.append(j)
                if len(keep) > 1 and np.hypot(*(pts[-1] - pts[keep[-1]])) < min_seg:
                    keep.pop()
                keep.append(len(pts) - 1)
                pts = pts[keep]
            out.append((z, LineString(pts)))
        return out

    def _clean_inset(self, poly):
        """Inset the outline by half a line width, then morphologically close
        it (dilate + erode by half a bead) to seal crevices narrower than the
        bead. Without this, deep folds make the offset boundary double back on
        itself — hairpin reversals the nozzle can't print (the factory Cura
        files contain zero of these)."""
        r = self.line_width / 2.0
        rc = getattr(self, "_close_r", self.line_width / 4.0)
        # inset, then closing (dilate + erode by rc): seal narrow folds
        inset = offset2d.offset_close(poly, -r, rc)
        if inset.is_empty:
            return None
        return inset.simplify(0.05)

    # Windows whose turning is within this much of the flattest one count as
    # equally flat (radians over the window, plus a share of the minimum).
    # On a round outline every window turns about the same: on the first
    # ring of Coil Bowl, Belly Vase and cyl60 all 64 windows sit within
    # 0.02 rad of each other, so the old argmin picked whichever came first
    # in the array by 0.0002 rad of float noise. Kuksa-v1 even has 4 exact
    # ties. 1.1 deg plus 5% of the minimum is far below anything a bead can
    # show, and wide enough that vertex noise cannot decide the seam.
    _SEAM_TIE_RAD = 0.02
    _SEAM_TIE_REL = 0.05
    # Among equally flat windows, starts whose Y lies within this of the
    # highest are tied too, and the lowest X wins.
    _SEAM_TIE_Y = 0.01

    @classmethod
    def _seam_to_flattest(cls, ring, window=5):
        """Rotate a closed ring so it starts (and therefore seams) on the
        flattest stretch of the outline, measured as the smallest total turning
        angle over a sliding window of vertices.

        Ties are settled by geometry, never by array order: of the windows
        that are equally flat (within _SEAM_TIE_RAD / _SEAM_TIE_REL of the
        flattest), the one starting at the highest Y wins, and among starts level
        within _SEAM_TIE_Y the one with the lowest X. Where a ring happens to start, or which of two near-equal
        windows is a hair flatter, used to decide the seam, so a sub-micron
        change anywhere upstream could carry it across the vase, and every
        turn above it with it: v1.0.12 fed Kuksa-v1 moved by 0.5 um moved its
        own turns by up to 2.27 mm. Now the seam only moves if the outline
        itself does."""
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
        lo = float(score.min())
        cand = np.nonzero(score <= lo + cls._SEAM_TIE_RAD + cls._SEAM_TIE_REL * lo)[0]
        y = pts[cand, 1]
        cand = cand[y >= y.max() - cls._SEAM_TIE_Y]
        k = int(cand[np.argmin(pts[cand, 0])])
        pts = np.roll(pts, -k, axis=0)
        return LineString(np.vstack([pts, pts[:1]]))

    # Heights where a vase layer's section narrowed abruptly and stayed narrow
    # (see _pick_section). A tuple here so a slice that never picks a vase
    # section still reads as "none"; _pick_section swaps in a fresh list.
    pick_giveups = ()
    _pick_run = 0           # consecutive give-ups that agree with each other
    _pick_run_poly = None   # the section the last give-up rejected

    # How many consecutive layers must agree on an abrupt narrowing before it
    # is believed. trimesh's fragment glitches flicker with height, so one or
    # two disagreeing layers are repeated over; a bottle's neck is still there
    # three layers (about 2 mm at 0.6 mm layers) later.
    _PICK_GIVEUP_CAP = 3

    def _pick_section(self, polys, prev_poly, z_target):
        """Choose the section polygon for a vase layer, guarding against
        trimesh's silent polygonization failures (it can return only a small
        FRAGMENT of the real section with no error — following it teleports
        the wall sideways and tears a hole).

        A candidate is suspicious when its area collapses versus the previous
        layer while its centroid jumps: real tapers shrink in place. Suspicious
        layers are re-sectioned at nudged heights; None means unrecoverable
        (caller repeats the previous ring), up to _PICK_GIVEUP_CAP layers in a
        row, after which a stable, contained narrowing is accepted as real."""
        # Judge every polygon on its SHELL (the area inside its outer ring),
        # never on its area with holes. A vase only prints the outer ring, and
        # a hollow cup's floor-to-cavity change is not a collapse: on the
        # user's Kuksa-v1 the section goes from a 5650 mm2 floor disc to an
        # annulus of 1286 mm2 with holes but 5801 mm2 of shell. Measured with
        # holes, that read as ratio 0.23 (< 0.25, "collapse"), every nudge
        # agreed, and since a give-up leaves prev_poly on the floor disc, 21
        # layers repeated the foot ring 8 mm inside the flaring wall before
        # smoothing ramped +3.3 mm per layer to catch up (more than a bead).
        # Judged on shell area: 0 repeated layers, steps of 0.23-0.96 mm, and
        # 128 mesh sections instead of 234. Solid models have no holes, so
        # their shell area IS their area and their output is unchanged.
        shell = lambda p: Polygon(p.exterior).area
        if prev_poly is None:
            # First vase layer of this slice: start the give-up bookkeeping.
            self.pick_giveups = []
            self._pick_run = 0
            self._pick_run_poly = None
        largest = max(polys, key=shell)
        if prev_poly is None:
            return largest
        prev_shell = shell(prev_poly)

        def suspicious(p):
            ratio = shell(p) / max(prev_shell, 1e-9)
            c0, c1 = prev_poly.centroid, p.centroid
            shift = math.hypot(c1.x - c0.x, c1.y - c0.y)
            drop_and_move = ratio < 0.6 and shift > 2.0 * self.line_width
            collapse = ratio < 0.25 and prev_shell > 200.0
            return drop_and_move or collapse

        if not suspicious(largest):
            self._pick_run = 0
            return largest
        # Re-section around the height the layer was actually sampled at
        # (the middle of its slab, z_target - lh/2, as slice() does). The old
        # fixed z_target - 0.1 centred them near the top of the slab: at
        # 0.6 mm layers the highest nudge sampled above the NEXT layer's own
        # height, and on a 2.7 mm layer every nudge landed within 0.4 mm of
        # the slab top, about a millimetre from where the rejected section
        # was taken.
        z_mid = z_target - self.layer_height / 2.0
        for dz in (0.15, -0.15, 0.3, -0.3, 0.45):
            alt = self._section_polygons(max(z_mid + dz, 1e-3))
            if not alt:
                continue
            cand = max(alt, key=shell)
            if not suspicious(cand):
                self._pick_run = 0
                return cand

        # Give-up. prev_poly is deliberately NOT moved to the rejected
        # section: if it were, one real fragment would become the reference
        # and the guard would follow it (the Paper Bag teleport it exists to
        # stop). Instead, count how many layers IN A ROW come back with the
        # same narrow section. A glitch changes with height; a bottle neck or
        # a lid knob does not, and repeating the body ring over it forever
        # printed a 40 mm phantom cylinder past the shoulder (bottle with a
        # flat shoulder: 66 repeated layers, verdict PASS).
        last = self._pick_run_poly
        if last is not None and self._pick_run > 0:
            r = shell(largest) / max(shell(last), 1e-9)
            c0, c1 = last.centroid, largest.centroid
            same = 0.5 <= r <= 2.0 and math.hypot(c1.x - c0.x, c1.y - c0.y) <= 2.0 * self.line_width
        else:
            same = False
        self._pick_run = self._pick_run + 1 if same else 1
        self._pick_run_poly = largest
        if self._pick_run < self._PICK_GIVEUP_CAP:
            return None
        if self._pick_run == self._PICK_GIVEUP_CAP:
            self.pick_giveups = [*self.pick_giveups, round(float(z_target), 3)]
            # Accept only a narrowing that sits INSIDE the old outline, the
            # way a neck sits on its shoulder. A section off to the side is a
            # different body (two towers, the narrow one outlasting the wide
            # one): following it would drag a bead through the air between
            # them, so those keep repeating the old ring (and stay recorded
            # above for the warning). The containment test is plain numpy
            # (even-odd ray casting against the old outline's vertices), so
            # this path never calls into GEOS at all and cannot raise a GEOS
            # error under WebAssembly, where those are fatal.
            pts = np.asarray(largest.exterior.coords)[:, :2]
            pts = pts[:: max(1, len(pts) // 64)]
            if float(np.mean(_points_in_ring(pts, np.asarray(prev_poly.exterior.coords)[:, :2]))) >= 0.9:
                self._pick_run = 0
                self._pick_run_poly = None
                return largest
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
            # inset, then seal sub-bead crevices (offset-boundary hairpins)
            inset = offset2d.offset_close(poly, -r, r_close).simplify(0.05)
            if inset.is_empty:
                continue
            pieces = sorted(inset.geoms if isinstance(inset, MultiPolygon) else [inset],
                            key=lambda g: g.area, reverse=True)
            if factor > 0 and not self._ring_is_whole(pieces, poly):
                continue   # would drop part of the wall -> retry with less inset
            return pieces[0].exterior
        return poly.exterior

    @staticmethod
    def _ring_is_whole(pieces, poly):
        """Does the largest inset piece stand for the whole wall?

        Three ways it does not, each relaxing the inset for that layer:
          * a second piece over 5 mm^2 is a real lobe the ring would drop
            (Paper Bag Vase folds);
          * the largest piece holds under 3/4 of the inset. A rim thinner
            than the bead splits into a necklace of near-equal slivers, and
            "the largest" is then just one bead of it: Bubble Vase's top
            layer came out as 13 pieces of 2.4 mm^2, so the last turn left
            the rim and jumped 30-60 mm across the vase to wherever that
            sliver sat (v1.0.12 did the same);
          * it is under 1% of the section's area: the inset collapsed to a
            speck (3DBenchy: 1.4 mm^2 left of a 162 mm^2 section), which is a
            point, not a wall.
        A normal inset is one piece with most of the section's area. Across
        the 30 tier-1 and synthetic models at defaults, the last two rules
        change exactly those two layers (Bubble Vase, 3DBenchy) and nothing
        else."""
        keep = pieces[0].area
        total = sum(g.area for g in pieces)
        if len(pieces) > 1 and pieces[1].area > 5.0:
            return False
        if keep < 0.75 * total:
            return False
        return keep >= 0.01 * poly.area

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

        Self-intersecting rings are repaired in offset2d (integer Clipper),
        not with buffer(0): GEOS buffer can abort the whole WebAssembly
        runtime (see offset2d). The repair keeps the largest piece, which is
        what buffer(0) + "largest" did here before.
        """
        polys = []
        for pts in rings:
            if len(pts) < 4:
                continue
            p = offset2d.repair_ring(pts)
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
                poly = offset2d.repair_polygon(polys[i].exterior.coords, holes)
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
                #
                # polygons_full "repairs" any invalid ring with GEOS buffer(),
                # which aborts the WebAssembly runtime outright on crumpled
                # outlines (Taco Bell Bag died there in Pyodide). So it only
                # gets rings that are already valid; a section with any
                # self-intersecting ring goes straight to our builder, which
                # repairs rings with Clipper instead. A section whose rings
                # are all valid takes exactly the old path.
                rings = path2d.discrete
                if all(Polygon(r).is_valid for r in rings if len(r) >= 4):
                    try:
                        polys = list(path2d.polygons_full)
                    except Exception as exc:
                        self._last_section_error = exc
                        self.used_ring_fallback = True
                        polys = self._rings_to_polygons(rings)
                else:
                    self.invalid_ring_sections = getattr(self, "invalid_ring_sections", 0) + 1
                    polys = self._rings_to_polygons(rings)
                if not polys:
                    continue
                return [affine_transform(p, [a, b, d, e, xoff, yoff])
                        for p in polys]
            except Exception as exc:
                self._last_section_error = exc
                continue
        self.section_failures = getattr(self, "section_failures", 0) + 1
        return []

    def _measure_support(self, layers):
        """Record the least-supported layer: the fraction of a layer's bead
        footprint that lands on clay laid down by the layer below.

        Footprints, their union and the overlap are all done in Clipper's
        integer space (offset2d.bead_bands), not with GEOS. This used to be
        LineString.buffer() plus union_all / intersection on a 1 um grid. The
        grid made the union and intersection robust, but buffer() itself
        always makes its first attempt at full float precision, and under
        WebAssembly a GEOS exception there aborts the whole runtime (see
        offset2d). It is the same 1 um grid, so the fractions agree with the
        GEOS ones to within a few thousandths."""
        half = self.line_width / 2.0
        prev_clay = None
        for lay in sorted(layers, key=lambda l: l["z"]):
            paths = [pp for pp in lay["paths"] if pp is not None and len(pp.coords) >= 2]
            clay = offset2d.bead_bands(paths, half)
            if not clay:
                continue
            area = offset2d.paths_area(clay)
            if prev_clay is not None and area > 0:
                frac = offset2d.intersection_area(clay, prev_clay) / area
                if frac < self.min_support_frac:
                    self.min_support_frac = float(frac)
                    self.min_support_z = float(lay["z"])
            prev_clay = clay

    def _concentric_fill(self, polygons, layer_index, bottom_layers, staggered,
                         offset_factor):
        paths = []
        step = self.line_width
        # Count parity from the TOP: the last base layer (index bottom_layers-1)
        # must never be inset, because the wall lands on its outermost ring.
        from_top = bottom_layers - 1 - layer_index
        offset = offset_factor * self.line_width if (staggered and from_top % 2 == 1) else 0.0
        for poly in polygons:
            # The OUTERMOST ring always sits on the model outline. Staggering
            # used to shift the whole layer inward, including this ring, which
            # on a model that flares outward left the next layer's outer ring
            # hanging over almost nothing: on a flared cup the top base layer
            # had 9% of its bead supported, so the rim of the base drooped and
            # the wall built on it never bonded. The stagger now shifts only
            # the interior rings, and the neighbouring layer covers the gap
            # that leaves, which is the whole point of staggering.
            d = 0.0
            outermost = True
            while True:
                buffered = offset2d.offset(poly, -d - self.line_width / 2)
                if buffered.is_empty or buffered.area < self.line_width ** 2:
                    break
                geoms = buffered.geoms if isinstance(buffered, MultiPolygon) else [buffered]
                for g in geoms:
                    paths.append(self._normalize_ring(g.exterior))
                    paths.extend(self._normalize_ring(i) for i in g.interiors)
                d += step + (offset if outermost else 0.0)
                outermost = False
        return paths

    # Base rings closer than this many bead widths are joined by an
    # extruding step-over instead of a stop. Neighbouring rings sit one bead
    # apart, and the first ring in on a staggered (inset) layer sits 1.5
    # beads in, so 1.6 catches that gap too, while a hole or a separate
    # island, which is further away, still gets its retract.
    STEP_OVER_BEADS = 1.6

    @staticmethod
    def _chain_base_rings(paths, reach=None, trim=0.0):
        """Roll each closed base ring so it starts right beside where the
        previous ring ended.

        buffer() starts every concentric ring wherever GEOS likes, often on
        the far side of the part, so the nozzle had to stop, retract and
        travel to reach it. Starting each ring next to the end of the one
        outside it makes the hop about one bead long, short enough for
        to_gcode to bridge with an extruding step-over instead of a stop.

        Simply taking the vertex nearest the previous end is not enough on a
        creased outline. A closed ring ends where it starts, and a start in
        the mouth of a fold can sit 1.7-2.2 beads from the ring inside it,
        because the inner ring rounds the fold off. Measured at defaults,
        nearest-start left 21 E-only stops per print on Paper Bag Vase and 15
        on Taco Bell Bag; this way both have 5, the same as the round
        samples (the opening prime and two per layer change). So the starts
        of a run of rings are chosen together: the fewest hops longer than
        `reach` (the longest hop to_gcode will bridge) first, then the
        shortest total hop length, which keeps the clay laid across the gaps
        to a minimum. That is a shortest path through one choice of start
        vertex per ring (a small dynamic programme, |ring| x |next ring|
        distances per step). The first ring of a layer may start anywhere,
        since the layer change travels to it anyway. Rings that cannot be
        reached within `reach` at all (a hole, another island) still get
        their retract, but at the shortest travel. On round parts this picks
        the same vertices as simply taking the nearest one would.

        A ring that hands over to the next one by a step-over also stops
        `trim` (half a bead) short of closing. The step-over lays clay over
        a gap the two beads already cover, about one bead of extra per ring,
        so the smaller the base the bigger the share: +3.0% on Coil Bowl,
        +3.6% on Spiral Vase's foot and +4.3% on the 40 mm + 20 mm test
        discs, past the +3.5% budget. The last half bead of a ring is laid
        on top of the round blob it started with, so it is mostly doubled
        clay too; dropping it turns the hop into a diagonal (about 3.4 mm
        instead of 3.0) and takes those to +1.8%, +2.3% and +2.5%. The head
        also turns about 63 degrees into the diagonal instead of 90, which
        cleared the remaining classic-jerk base hitches on Coil Bowl (4 -> 0)
        and the test discs (9 -> 1). A ring is only trimmed when the
        diagonal still fits inside `reach`, or it would lose its end and
        get a retract anyway.

        Runs after _resample, so every ring keeps exactly the vertices it had
        (only the start moves, and a trimmed ring ends on one interpolated
        point). Pure numpy, no shapely overlay, so nothing here can reach
        GEOS under WebAssembly. Path order is kept: outer to inner, holes
        after their exterior. Open paths are left alone."""
        over = 1e6   # cost of one hop the step-over cannot bridge

        def closed_pts(p):
            if p is None:
                return None
            c = np.asarray(p.coords)
            if len(c) > 3 and np.allclose(c[0], c[-1]):
                return c[:-1]
            return None

        def hop_cost(d):
            return d if reach is None else d + over * (d > reach)

        def trimmed(c, nxt):
            # c: closed ring (first == last). End it `trim` short, if the
            # diagonal from there to the next ring's start fits in reach.
            seg = np.hypot(*np.diff(c, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            target = cum[-1] - trim
            if cum[-1] < 4 * trim or math.hypot(*(c[-1] - nxt)) > reach:
                return None
            m = int(np.searchsorted(cum, target))   # cum[m-1] < target <= cum[m]
            f = (target - cum[m - 1]) / seg[m - 1]
            end = c[m - 1] + (c[m] - c[m - 1]) * f
            if math.hypot(*(end - nxt)) > reach:
                return None
            if f < 1e-6:
                return c[:m]
            return np.vstack([c[:m], end])

        # Only the paths to_gcode will draw: it skips None and paths of
        # fewer than two points, so those must not split a run of rings.
        drawn = [k for k, p in enumerate(paths)
                 if p is not None and len(p.coords) >= 2]
        rings = [closed_pts(paths[k]) for k in drawn]
        out = list(paths)
        last = None      # where the previous path ended
        i, n = 0, len(drawn)
        while i < n:
            if rings[i] is None:
                last = np.asarray(paths[drawn[i]].coords)[-1]
                i += 1
                continue
            j = i
            while j < n and rings[j] is not None:
                j += 1
            run = rings[i:j]
            first = run[0]
            if last is None:
                cost = np.zeros(len(first))
            else:
                cost = hop_cost(np.hypot(first[:, 0] - last[0], first[:, 1] - last[1]))
            back = []
            for k in range(1, len(run)):
                A, B = run[k - 1], run[k]
                best = np.full(len(B), np.inf)
                arg = np.zeros(len(B), dtype=np.int64)
                # Rows in chunks so a long ring never builds a huge matrix:
                # about 500k distances (4 MB per temporary) at a time, which
                # the browser's WebAssembly heap takes in its stride. A
                # 200 mm base ring (about 630 points) still goes in one
                # chunk. Ties keep the first index, so chunking never
                # changes the answer.
                rows = max(1, 500000 // len(B))
                for r0 in range(0, len(A), rows):
                    a = A[r0:r0 + rows]
                    d = np.hypot(a[:, 0][:, None] - B[:, 0][None, :],
                                 a[:, 1][:, None] - B[:, 1][None, :])
                    tot = cost[r0:r0 + rows][:, None] + hop_cost(d)
                    am = np.argmin(tot, axis=0)
                    v = tot[am, np.arange(len(B))]
                    better = v < best
                    best[better] = v[better]
                    arg[better] = am[better] + r0
                cost = best
                back.append(arg)
            s = int(np.argmin(cost))
            starts = [s]
            for arg in reversed(back):
                s = int(arg[s])
                starts.append(s)
            starts.reverse()
            for k, s in enumerate(starts):
                pts = np.roll(run[k], -s, axis=0)
                c = np.vstack([pts, pts[:1]])
                cut = None
                if trim > 0 and reach is not None and k + 1 < len(run):
                    cut = trimmed(c, run[k + 1][starts[k + 1]])
                if cut is not None:
                    out[drawn[i + k]] = LineString(cut)
                elif s:
                    out[drawn[i + k]] = LineString(c)
            last = run[-1][starts[-1]]
            i = j
        return out

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
                 stagger_fill=1.0, base_flow=1.0, staggered=None):
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
        g += [f";Layer height: {self.layer_height:g}"]
        if abs(self.base_layer_height - self.layer_height) > 1e-9:
            g.append(f";Base layer height: {self.base_layer_height:g}")
        g.append(f";Line width: {self.line_width:g}")
        # Record the stagger setting. A validator cannot reliably infer it
        # from the toolpath: it has to compare ring radii, and on a base
        # that is not round those vary more within one ring than the
        # stagger shifts them between layers. Reading it back beats
        # guessing, and telling someone to switch on what is already on is
        # worse than saying nothing.
        if staggered is not None:
            g.append(f";Staggered base: {1 if staggered else 0}")
        maxz_at = None
        if xs:
            g += [f";MINX:{min(xs):.2f}", f";MINY:{min(ys):.2f}", f";MINZ:{min(zs):.2f}",
                  f";MAXX:{max(xs):.2f}", f";MAXY:{max(ys):.2f}", f";MAXZ:{max(zs):.2f}"]
            # Filled in after the toolpath is written. A vase layer climbs a
            # whole layer height above its own z, so the layer list's top z
            # was one layer short of where the nozzle really goes (Coil Bowl
            # said 61.80, the file reaches 62.40).
            maxz_at = len(g) - 1
        g.append(f";LAYER_COUNT:{len(layers)}")
        g.append(profile["start_gcode"])
        g.append("M107")

        total_e = 0.0
        first = True
        last_xy = None    # nozzle position after the previous path (for joins)
        last_z = None     # nozzle Z after the previous path
        last_type = None  # layer type of the previous path ("bottom"/"vase")
        max_z = None      # highest Z actually emitted, for ;MAXZ
        step_over_max = self.STEP_OVER_BEADS * self.line_width
        for li, layer in enumerate(layers):
            g.append(f";LAYER:{li}")
            z = layer["z"]
            is_vase = layer["type"] == "vase"
            if li == 0:
                h = self.first_layer_height
            elif layer["type"] == "bottom":
                h = self.base_layer_height
            else:
                h = self.layer_height
            layer_e = (h * self.line_width / self.filament_area) \
                * (first_layer_flow if li == 0 else 1.0)
            if layer["type"] == "bottom":
                # The solid base is filled at exactly one bead width per
                # pass, so on a wide footprint it lays down a lot of clay,
                # and clay spreads under its own weight far more than
                # plastic. This trims the base only; walls are untouched.
                layer_e *= base_flow
            if layer.get("inset"):
                layer_e *= stagger_fill
            g.append(";TYPE:SKIN" if layer["type"] == "bottom" else ";TYPE:WALL-OUTER")

            in_layer = False   # has this layer drawn a path yet?
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
                elif (continuous and is_vase and pi == 0 and last_type == "vase"
                        and join_d is not None and join_d < 2.0):
                    # Continuous spiral: extrude across the tiny seam jump —
                    # the bead never breaks between layers. When the turn
                    # already ends on the next start (the v1.0.11 seam), the
                    # join is a zero-length G1 with E+0, nothing for the
                    # printer to do, so leave it out. Compared as written,
                    # not as floats: a 0.0001 mm gap still prints the same
                    # line twice (cyl60 had 2). The E still counts, so every
                    # later E value is the same as before.
                    # Only from a vase turn: the base-to-wall step keeps its
                    # retract and single G0 even when the base happens to end
                    # within 2 mm of the wall start. On the wp/C branch
                    # (Clipper offsets, other ring starts) cube_corner's base
                    # ended 1.1 mm from it, and the join climbed a whole layer
                    # (Z 3.0 -> 3.6) in one short move, a 0.58 mm kink in the
                    # first turn's arc-length ramp; no v1.0.12 slice in the
                    # 336-case suite took this path.
                    prev = (f"G1 X{last_xy[0]:.3f} Y{last_xy[1]:.3f} Z{last_z:.3f} "
                            f"E{total_e:.5f}")
                    total_e += join_d * layer_e
                    line = f"G1 X{sx:.3f} Y{sy:.3f} Z{z:.3f} E{total_e:.5f}"
                    if line != prev:
                        g.append(line)
                elif (layer["type"] == "bottom" and in_layer
                        and join_d is not None and join_d <= step_over_max):
                    # Next base ring, about one bead in (_chain_base_rings
                    # starts it beside this ring's end): keep extruding
                    # across instead of retract / travel / unretract. Those
                    # two E-only moves bring the head to a dead stop, 32-104
                    # times per print on the samples against 3 in Eazao's
                    # own Bowl.gcode. The hop must extrude at full flow: a
                    # dry G1 still stops under XYZE junction deviation (19-37
                    # stops left on the Eazao-JD model), because E stopping
                    # is a corner too, and half flow traded the stops for
                    # 16-23 classic-jerk hitches per print (Coil Bowl, Spiral
                    # Vase, Twist Pot). Extra clay laid over the gap, with
                    # the half-bead trim: +0.8% (Cuboid, Paper Bag) to +2.5%
                    # (the small test discs) of the base. Layer changes and the
                    # base-to-wall move keep their retract.
                    total_e += join_d * layer_e
                    g.append(f"G1 X{sx:.3f} Y{sy:.3f} E{total_e:.5f}")
                else:
                    # Travel with a short retract to limit ooze (clay can't
                    # do big retracts).
                    total_e -= 0.1
                    g.append(f"G1 F{f_print} E{total_e:.5f}")
                    g.append(f"G0 F{f_print} X{sx:.3f} Y{sy:.3f} Z{z:.3f}")
                    total_e += 0.1
                    g.append(f"G1 F{f_print} E{total_e:.5f}")

                n = len(coords) - 1
                seg = [math.hypot(coords[j + 1][0] - coords[j][0],
                                  coords[j + 1][1] - coords[j][1]) for j in range(n)]
                if is_vase:
                    # Spread the climb by distance travelled, not by point
                    # count. Per point, a 0.05 mm segment at the seam got the
                    # same Z step as a 1.5 mm one, a Z slope 30x steeper than
                    # the wall around it (up to 4000x on collapsed points),
                    # and Eazao's 0.3 mm/s Z jerk braked the head to 1-4 mm/s
                    # there: 25-68 brakes per pot at defaults, none in the
                    # factory Bowl. By arc length the slope is the same all
                    # the way round. XY and E are untouched, and the last
                    # point still lands at exactly z + layer height, so the
                    # turn ends on the next turn's start as before.
                    cum = []
                    run = 0.0
                    for s in seg:
                        run += s
                        cum.append(run)
                    tot = run
                for j in range(n):
                    x2, y2 = coords[j + 1]
                    total_e += seg[j] * layer_e
                    if is_vase:
                        frac = cum[j] / tot if tot > 1e-9 else (j + 1) / n
                        zj = z + frac * self.layer_height
                        g.append(f"G1 X{x2:.3f} Y{y2:.3f} Z{zj:.3f} E{total_e:.5f}")
                    else:
                        g.append(f"G1 X{x2:.3f} Y{y2:.3f} E{total_e:.5f}")
                last_xy = coords[-1]
                last_z = zj if is_vase else z
                last_type = layer["type"]
                if max_z is None or last_z > max_z:
                    max_z = last_z
                in_layer = True

        if maxz_at is not None and max_z is not None:
            g[maxz_at] = f";MAXZ:{max_z:.2f}"
        g.append(profile["end_gcode"])
        return "\n".join(g)


def slice_stl(stl_path, profile, nozzle=3.0, layer_height=1.0, bottom_layers=3,
              staggered=True, staggered_offset_factor=0.5, vase_mode=True,
              path_resolution=1.5, line_width=None, first_layer_flow=1.0,
              source=None, first_layer_height=None, continuous=True,
              fold_softening=None, scale=1.0, stagger_fill=1.0, base_flow=1.0,
              base_layer_height=None, diagnostics=None):
    """Convenience wrapper: returns (gcode_str, layers) for preview + export.

    diagnostics: optional dict, filled with how many heights failed to section
    and why — so the UI can tell the user when a model only partly sliced
    instead of silently handing back a stump.
    """
    slicer = STLSlicer(stl_path, profile, nozzle=nozzle, layer_height=layer_height,
                       line_width=line_width, first_layer_height=first_layer_height,
                       scale=scale, base_layer_height=base_layer_height)
    layers = slicer.slice(bottom_layers, staggered, staggered_offset_factor,
                          vase_mode, path_resolution, fold_softening=fold_softening,
                          spiral=continuous)
    if diagnostics is not None:
        z_max = slicer.mesh.bounds[1][2]
        diagnostics["failed_heights"] = getattr(slicer, "section_failures", 0)
        diagnostics["expected_heights"] = max(1, int(z_max / max(layer_height, 1e-6)))
        diagnostics["layers"] = len(layers)
        err = getattr(slicer, "_last_section_error", None)
        diagnostics["last_error"] = f"{type(err).__name__}: {err}" if err else None
        diagnostics["model_top_mm"] = float(z_max)
        diagnostics["min_support"] = float(getattr(slicer, "min_support_frac", 1.0))
        diagnostics["min_support_z"] = getattr(slicer, "min_support_z", None)
        # Mesh quality. A model with holes or flipped faces sections
        # unpredictably, which shows up as missing or wandering layers.
        try:
            diagnostics["watertight"] = bool(slicer.mesh.is_watertight)
            diagnostics["winding_ok"] = bool(slicer.mesh.is_winding_consistent)
        except Exception:
            diagnostics["watertight"] = None
            diagnostics["winding_ok"] = None
        diagnostics["sliced_top_mm"] = float(max((l["z"] for l in layers), default=0.0))
        # Pre-checks the UI turns into warnings: will it fit the bed (and the
        # scale that would make it), does it start above the bed, where the
        # section picker gave up on an abrupt narrowing, and how many layers
        # lost a detached island (that last one is filled in a later release).
        diagnostics["fits_bed"] = slicer.fits_bed
        diagnostics["fit_scale"] = slicer.fit_scale
        diagnostics["floating_start"] = slicer.floating_start
        diagnostics["empty_layers_dropped"] = getattr(slicer, "empty_layers_dropped", 0)
        diagnostics["pick_giveups"] = list(getattr(slicer, "pick_giveups", []))
        diagnostics["dropped_island_layers"] = getattr(slicer, "dropped_island_layers", 0)
    return slicer.to_gcode(layers, first_layer_flow=first_layer_flow, source=source,
                           continuous=continuous, stagger_fill=stagger_fill,
                           base_flow=base_flow, staggered=staggered), layers
