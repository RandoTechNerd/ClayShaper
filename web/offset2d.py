"""
Polygon offsets and ring repair for the slicer, done with Clipper (integer
arithmetic) instead of GEOS buffer().

Why: in the browser build, GEOS buffer() can take down the whole Python
runtime. Pyodide's GEOS (3.12.1) throws a C++ TopologyException on some
near-degenerate but perfectly valid outlines (folds, crumpled walls), and
under WebAssembly that exception is NOT catchable: GEOS's own retry-at-lower-
precision path never runs, Python's try/except never sees it, and the page
just freezes ("Pyodide has suffered a fatal error"). Measured in the real app
on 5 of 26 real models, including Eazao's own Taco Bell Bag, Bubble Vase and
Paper Bag Vase; Bulldog died at the upload thumbnail before Slice was pressed.
Desktop Python (GEOS 3.13) slices all 26 without a murmur, so only the
browser ever shows it.

Snapping the input first does not help: make_valid + set_precision on a
0.01 mm grid before buffer() still aborted Bubble Vase in the app (the failing
point sat exactly on the grid). GEOS's buffer always makes its first attempt
at full float precision and only retries on a snapped grid after CATCHING the
exception, and that catch is exactly what never happens under WebAssembly.
Clipper works on integer coordinates, so it has no such failure path, and it
ships in Pyodide's own package set (pyclipper), so it costs no extra host.

Settings are fixed so results repeat exactly run to run and desktop matches
browser:
  * 1 um integer grid (SCALE). Far finer than any clay bead.
  * Round joins with the same arc density GEOS used here by default
    (16 segments per quarter circle, i.e. buffer's quad_segs=16), so the
    outlines keep the shape every sample model was tuned against.
  * A 2 um vertex clean-up before each offset (_CLEAN_UM, below).
Against the GEOS engine on the round samples (Coil Bowl, Twist Pot, Belly
Vase, Cuboid, Eazao Bowl): every vase ring within 0.071 mm before smoothing
(both sides go through simplify(0.05)), base ring counts identical, and
min_support within 0.004. GEOS also left zero-area sliver rings now and
then (one became a 16th base ring on Paper Bag Vase); Clipper does not.

Shapely is still used for everything that is not an overlay (area, simplify,
centroid, containment), and every polygon handed back is a normal shapely
Polygon / MultiPolygon, so callers did not change shape.
"""

import math

import numpy as np
import pyclipper
from shapely.geometry import Polygon, MultiPolygon

SCALE = 1000.0                  # mm -> integer um
_ARC = 1.0 - math.cos(math.pi / 64.0)   # sagitta per unit radius at quad_segs=16
_MIN_AREA = 1e-9                # mm^2: drop only true zero-area slivers
# Vertices closer than this to the line of their neighbours are dropped before
# offsetting (Clipper's CleanPolygons). trimesh sections of dense meshes are
# full of near-duplicate points (Paper Bag Vase: 1% of segments are under
# 4 um), and every one of them is a join Clipper has to round and union.
# At 2 um it halves the vertex count there and makes an offset 3x faster
# (17.9 -> 6.1 ms a section, level with GEOS at 6.0), while the result stays
# within 0.010 mm of GEOS's outline. 3 um already moved it 0.024 mm.
_CLEAN_UM = 2


def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def _path(coords, outer):
    """Scale a shapely ring to integer um, oriented the way Clipper expects
    (outer counter-clockwise, holes clockwise). None for a ring with fewer
    than 3 distinct points. A self-crossing ring (a bow-tie) can have zero
    net area and still enclose clay, so it is kept as given."""
    pts = np.rint(np.asarray(coords, dtype=float)[:, :2] * SCALE).astype(np.int64)
    if len(pts) > 1 and (pts[0] == pts[-1]).all():
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    area2 = int(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))   # exact in int64
    if area2 == 0:
        if len(np.unique(pts, axis=0)) < 3:
            return None
    elif (area2 > 0) != outer:
        pts = pts[::-1]
    return pts.tolist()


def _paths(geom):
    out = []
    for g in _polys(geom):
        p = _path(g.exterior.coords, True)
        if p is None:
            continue
        out.append(p)
        for r in g.interiors:
            h = _path(r.coords, False)
            if h is not None:
                out.append(h)
    return out


def _ring(contour):
    """Clipper contour -> float mm ring, rolled to start at its leftmost
    vertex (lowest x, then lowest y).

    Where a ring starts is not just cosmetic here: simplify() always keeps
    a ring's first vertex, so it decides which vertices survive. Clipper
    starts its rings just after the lowest point; GEOS's buffer started them
    near the leftmost one (it builds rings outward from its lowest-x node).
    Rolling to the leftmost vertex keeps the simplified outlines close to
    the GEOS ones. (It also used to steer the seam, which took the first of
    several equally flat stretches; _seam_to_flattest now settles those ties
    by geometry, so the start no longer matters there.) Median per-turn
    drift against the GEOS slices, without -> with the roll, measured
    before that seam change: Eazao Bowl 0.180 -> 0.008 mm, Coil Bowl 0.265
    -> 0.197, Twist Pot 0.521 -> 0.502; the other round samples moved by
    0.01 mm or less."""
    c = np.asarray(contour, dtype=np.int64)
    k = int(np.lexsort((c[:, 1], c[:, 0]))[0])
    return np.roll(c, -k, axis=0).astype(float) / SCALE


def _from_tree(node, acc):
    """Clipper PolyTree -> shapely polygons. Children of the root are outer
    contours, their children are holes, and a hole's children are islands
    sitting inside that hole (handled by recursing)."""
    for outer in node.Childs:
        if len(outer.Contour) < 3:
            continue
        holes = [_ring(h.Contour) for h in outer.Childs if len(h.Contour) >= 3]
        poly = Polygon(_ring(outer.Contour), holes)
        if poly.area > _MIN_AREA:
            acc.append(poly)
        for h in outer.Childs:
            _from_tree(h, acc)
    return acc


def _pack(polys):
    if not polys:
        return Polygon()
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _chain(geom, distances):
    """Offset by each distance in turn (mm), staying in Clipper's integer
    space between steps: one conversion in and one out, however many steps
    (a closing is two). Intermediate results come back from Clipper with
    outers counter-clockwise and holes clockwise, exactly what the next
    offset expects. Empty Polygon as soon as a step leaves nothing."""
    paths = _paths(geom)
    for k, d in enumerate(distances):
        paths = [p for p in pyclipper.CleanPolygons(paths, _CLEAN_UM) if len(p) >= 3]
        if not paths:
            return Polygon()
        delta = d * SCALE
        pco = pyclipper.PyclipperOffset(arc_tolerance=max(abs(delta) * _ARC, 0.25))
        pco.AddPaths(paths, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        if k == len(distances) - 1:
            return _pack(_from_tree(pco.Execute2(delta), []))
        paths = pco.Execute(delta)
    return Polygon()


def offset(geom, d):
    """Drop-in for geom.buffer(d) on a Polygon / MultiPolygon: grow (d > 0)
    or shrink (d < 0) by d mm with round joins. Empty Polygon when nothing
    is left, a MultiPolygon when the offset splits the shape."""
    if d == 0:
        return union(geom)          # buffer(0) semantics: a clean union
    return _chain(geom, (d,))


def close(geom, r):
    """Morphological closing (grow by r, then shrink by r): seals crevices
    narrower than 2r without moving the rest of the outline. Same as the
    old buffer(r).buffer(-r) pair."""
    if r <= 0:
        return geom
    return _chain(geom, (r, -r))


def offset_close(geom, d, r):
    """offset(geom, d) followed by close(..., r), in one pass through
    Clipper (the slicer's inset-then-seal step, run on every layer)."""
    if r <= 0:
        return offset(geom, d)
    if d == 0:
        return close(geom, r)
    return _chain(geom, (d, r, -r))


def _clip(subject, clip=(), op=pyclipper.CT_UNION):
    pc = pyclipper.Pyclipper()
    # StrictlySimple: no ring may touch itself, so every polygon handed back
    # is valid for shapely's containment tests (an invalid polygon can make
    # GEOS relate() throw, which would be the same runtime abort again).
    pc.StrictlySimple = True
    try:
        # pyclipper raises a plain Python ClipperException (catchable, also
        # in the browser) when a path is degenerate, e.g. all points collinear.
        pc.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    except pyclipper.ClipperException:
        return []
    if clip:
        try:
            pc.AddPaths(list(clip), pyclipper.PT_CLIP, True)
        except pyclipper.ClipperException:
            pass
    return _from_tree(pc.Execute2(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO), [])


def union(geom):
    """Valid union of a (possibly self-intersecting) Polygon / MultiPolygon."""
    paths = _paths(geom)
    return _pack(_clip(paths)) if paths else Polygon()


def _largest_valid(polys):
    if not polys:
        return None
    best = max(polys, key=lambda g: g.area)
    if not best.is_valid:
        # Belt and braces: an outer contour from a strictly simple result is
        # a simple ring, so dropping the holes always gives a valid polygon.
        best = Polygon(best.exterior.coords)
    return best if best.is_valid and best.area > _MIN_AREA else None


def repair_ring(pts):
    """One closed section ring -> a valid Polygon, or None.

    A valid ring comes back untouched. A self-intersecting one is filled
    with the non-zero rule and its largest piece kept, which is what the
    old buffer(0) + "keep the largest" did, minus the abort risk."""
    p = Polygon(pts)
    if p.is_valid:
        return p
    path = _path(p.exterior.coords, True)
    if path is None:
        return None
    return _largest_valid(_clip([path]))


def repair_polygon(shell, holes):
    """Shell minus holes as a valid Polygon (largest piece), or None. Used
    when a nested outline + holes does not form a valid polygon as given
    (a hole touching or crossing its outline)."""
    s = _path(shell, True)
    if s is None:
        return None
    hs = [h for h in (_path(c, True) for c in holes) if h is not None]
    return _largest_valid(_clip([s], hs, pyclipper.CT_DIFFERENCE))


# --- Bead footprints (open-path offsets) -------------------------------------
# The overhang check in stl_slicer._measure_support needs the strip of clay
# each toolpath lays down: the path thickened by half a bead each side. That
# used to be GEOS LineString.buffer(), the last buffer() left in the slice
# path. It never crashed on the 5 models that abort the browser, but it is
# the same GEOS routine with the same uncatchable failure mode, so it moves
# to Clipper too, and the union and intersection behind the support fraction
# stay in Clipper's integer space with it (no shapely overlay at all). The
# footprints are only ever measured (an area), never printed.


def _line_path(coords):
    """A toolpath (any (n, 2+) coords) -> integer um points, consecutive
    duplicates dropped. None when fewer than 2 distinct points remain."""
    pts = np.rint(np.asarray(coords, dtype=float)[:, :2] * SCALE).astype(np.int64)
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]
    return pts if len(pts) >= 2 else None


def bead_bands(lines, half):
    """Union of the strips `half` mm either side of every toolpath in
    `lines` (shapely LineStrings or coordinate arrays), as integer Clipper
    paths. A path that closes on itself is offset as a closed line (an
    annulus, what buffer() gave for a closed LineString); an open one gets
    round caps, buffer()'s default. Same arc density as everywhere else in
    this module (buffer's quad_segs=16). One offset call per layer, and
    Clipper unions the strips as part of it. Empty list when nothing is
    left."""
    delta = half * SCALE
    pco = pyclipper.PyclipperOffset(arc_tolerance=max(abs(delta) * _ARC, 0.25))
    n = 0
    for ln in lines:
        if ln is None:
            continue
        coords = getattr(ln, "coords", ln)
        pts = _line_path(coords)
        if pts is None:
            continue
        closed = len(pts) > 3 and (pts[0] == pts[-1]).all()
        if closed:
            pts = pts[:-1]
        try:
            pco.AddPath(pts.tolist(), pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDLINE if closed else pyclipper.ET_OPENROUND)
            n += 1
        except pyclipper.ClipperException:
            continue        # degenerate path: catchable, also in the browser
    if not n:
        return []
    return pco.Execute(delta)


def paths_area(paths):
    """Net area (mm^2) of Clipper paths: outers count positive, holes
    negative, as Clipper orients them."""
    return sum(pyclipper.Area(p) for p in paths) / (SCALE * SCALE)


def intersection_area(a, b):
    """Area (mm^2) where two sets of Clipper paths overlap (non-zero fill)."""
    if not a or not b:
        return 0.0
    pc = pyclipper.Pyclipper()
    try:
        pc.AddPaths(a, pyclipper.PT_SUBJECT, True)
        pc.AddPaths(b, pyclipper.PT_CLIP, True)
    except pyclipper.ClipperException:
        return 0.0
    return paths_area(pc.Execute(pyclipper.CT_INTERSECTION,
                                 pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
