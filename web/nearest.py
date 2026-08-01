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
"""

import numpy as np


class NearestPoints:
    """Drop-in for scipy.spatial.cKDTree limited to the .query() we use."""

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
            raise ValueError("NearestPoints: empty point set")
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
