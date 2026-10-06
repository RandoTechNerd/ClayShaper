"""
Tiny numpy stand-ins for the two scipy features the slicer used.

Why: scipy is ~30 MB in the browser build (Pyodide) — by far the largest
download and the main reason a cold first visit took ~20 s. We only ever used
two things from it, both on small data:

  * cKDTree(pts).query(q) -> (distance, index) of the nearest stored point.
    Our point sets are capped at 720 points per contour, so the brute-force
    distance matrix is a few MB and runs in milliseconds.
  * ndimage.binary_dilation(grid, iterations=n) with the default 4-neighbour
    structure — three lines of boolean array shifting.

Both are exact drop-in replacements for how the engine calls them, so slicing
and validation results are unchanged.

Update (v1.0.14 review): the "keep scipy out" reason turned out to be gone.
index.html has to load scipy anyway, because trimesh.section() imports it to
chain section edges, so this stand-in saved no download at all, and its
brute-force distance matrix became the single biggest hotspot in a slice.
Profiled natively it was 10.4 s of 30.0 s on Octopus Vase (slice + validate)
and 18.2 s of 55.5 s on Paper Bag. In Pyodide 0.27.6 swapping in the real
cKDTree took Octopus from 66.8 s to 27.1 s and Twist Pot from 14.2 s to
10.3 s, with byte-identical G-code. So NearestPoints() now hands back
scipy's cKDTree when scipy imports, and only falls back to the numpy class
below when it does not (a desktop Python without scipy). Both return the
same nearest index; the regression harness checks the G-code SHA is
unchanged on every quick-suite case.
"""

import numpy as np

try:
    # Guarded on purpose: this is the only direct scipy import in the engine.
    # scipy ships as a prebuilt Pyodide wheel, but if it ever fails to load
    # the slicer must still work, just slower.
    from scipy.spatial import cKDTree as _cKDTree
except Exception:          # ImportError, or a broken wheel raising at import
    _cKDTree = None


def NearestPoints(pts):
    """Nearest-point index over `pts`: cKDTree when available, else numpy.

    Both expose .query(q) -> (distance, index) with cKDTree's shape rules.
    An empty point set is refused here, up front. cKDTree would otherwise
    build happily and then answer every query with distance inf and index
    len(pts) (one past the end), which the callers would use to index the
    ring and crash far away from the real cause, or silently read garbage.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        raise ValueError("NearestPoints: empty point set")
    if _cKDTree is not None:
        return _cKDTree(pts)
    return NumpyNearest(pts)


class NumpyNearest:
    """Drop-in for scipy.spatial.cKDTree limited to the .query() we use.

    Fallback for when scipy is missing (see NearestPoints above)."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=float)

    def query(self, q, chunk=256):
        """Nearest stored point for each query point.

        Returns (distances, indices), matching cKDTree.query's shape rules:
        a single (2,) query gives scalars, an (N,2) array gives (N,) arrays.
        Chunked so the pairwise matrix stays small in WebAssembly memory.
        """
        q = np.asarray(q, dtype=float)
        single = (q.ndim == 1)
        if single:
            q = q[None, :]
        n = len(q)
        dist = np.empty(n, dtype=float)
        idx = np.empty(n, dtype=np.intp)
        P = self.pts
        if len(P) == 0:
            raise ValueError("NumpyNearest: empty point set")
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            diff = q[s:e, None, :] - P[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            j = d2.argmin(axis=1)
            idx[s:e] = j
            dist[s:e] = np.sqrt(d2[np.arange(e - s), j])
        if single:
            return float(dist[0]), int(idx[0])
        return dist, idx


def binary_dilation(grid, iterations=1):
    """4-neighbour boolean dilation, zero-padded — matches scipy's default."""
    g = np.asarray(grid, dtype=bool)
    for _ in range(max(1, int(iterations))):
        out = g.copy()
        out[1:, :] |= g[:-1, :]
        out[:-1, :] |= g[1:, :]
        out[:, 1:] |= g[:, :-1]
        out[:, :-1] |= g[:, 1:]
        g = out
    return g
