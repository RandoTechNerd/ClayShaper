"""
Shaded 3D toolpath preview (Orca-style).

Instead of plotting the toolpath as flat lines, each continuous extrusion run
is turned into a vertical ribbon one layer tall and rendered as a lit Mesh3d:
the result reads as a solid printed object with visible layer banding, like a
desktop slicer's preview, rather than a wire blob.
"""

import numpy as np
import plotly.graph_objects as go


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def runs_to_mesh(runs, layer_height, color, name="", legend=False, overlap=1.25,
                 bead_width=3.0):
    """
    Build one lit Mesh3d from extrusion runs, rendered as CLOSED bead tubes.

    Each run becomes a diamond-profile tube around the path — four rings of
    vertices (bottom, outer, top, inner) connected into a sealed surface, like
    a slicer's bead rendering. Closed tubes have no open faces, so leaning
    walls and folds can't show see-through slats or torn cap edges the way
    flat ribbons + caps did.

    runs: list of (x, y, z) arrays — z is the bead's TOP height; the bead
          spans z-layer_height*overlap .. z (slight overlap seals stacking).
    """
    r, g, b = _hex_to_rgb(color) if isinstance(color, str) else color
    dark = f"rgb({max(r-46,0)},{max(g-40,0)},{max(b-34,0)})"
    mid = f"rgb({max(r-18,0)},{max(g-16,0)},{max(b-13,0)})"
    lite = f"rgb({min(r+14,255)},{min(g+12,255)},{min(b+10,255)})"

    xs, ys, zs, vc = [], [], [], []
    I, J, K = [], [], []
    off = 0
    half = bead_width / 2.0
    for x, y, z in runs:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        n = len(x)
        if n < 2:
            continue
        # 2D path normals (perpendicular to travel direction).
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.hypot(dx, dy)
        norm[norm < 1e-9] = 1.0
        nx = dy / norm
        ny = -dx / norm

        h = layer_height * overlap
        z_bot = np.maximum(z - h, 0.0)
        z_mid = np.maximum(z - h / 2.0, 0.0)

        # four vertex rings along the path: Bottom, Outer, Top, Inner
        xs.append(x);            ys.append(y);            zs.append(z_bot)   # B
        xs.append(x + nx * half); ys.append(y + ny * half); zs.append(z_mid)  # O
        xs.append(x);            ys.append(y);            zs.append(z)      # T
        xs.append(x - nx * half); ys.append(y - ny * half); zs.append(z_mid)  # I
        vc.extend([dark] * n)
        vc.extend([mid] * n)
        vc.extend([lite] * n)
        vc.extend([mid] * n)

        ar = np.arange(n - 1)
        ring = [off, off + n, off + 2 * n, off + 3 * n]   # B, O, T, I starts
        for a, bgn in ((0, 1), (1, 2), (2, 3), (3, 0)):   # B-O, O-T, T-I, I-B
            a0 = ring[a] + ar
            a1 = a0 + 1
            b0 = ring[bgn] + ar
            b1 = b0 + 1
            I.append(np.concatenate([a0, b0]))
            J.append(np.concatenate([a1, a1]))
            K.append(np.concatenate([b0, b1]))
        off += 4 * n

    if not xs:
        return None
    return go.Mesh3d(
        x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs),
        i=np.concatenate(I), j=np.concatenate(J), k=np.concatenate(K),
        vertexcolor=vc,
        lighting=dict(ambient=0.42, diffuse=0.75, specular=0.18,
                      roughness=0.55, fresnel=0.05),
        lightposition=dict(x=200, y=100, z=400),
        flatshading=False,
        name=name, showlegend=legend,
        hoverinfo="skip",
    )


def bead_mesh_arrays(x, y, z, layer_height, color, bead_width=3.0, overlap=1.25):
    """
    Precompute bead-tube mesh arrays for PROGRESSIVE reveal (Print Simulator).

    Same closed diamond-profile tube as runs_to_mesh, but laid out for cheap
    partial rendering: vertices are interleaved 4-per-path-point (B,O,T,I) and
    triangles are ordered by path segment (8 per segment), so showing the print
    up to point p is just verts[:4p] + tris[:8(p-1)] — two numpy slices per
    frame instead of rebuilding any geometry.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)
    if n < 2:
        return None
    r, g, b = _hex_to_rgb(color) if isinstance(color, str) else color
    dark = f"rgb({max(r-46,0)},{max(g-40,0)},{max(b-34,0)})"
    mid = f"rgb({max(r-18,0)},{max(g-16,0)},{max(b-13,0)})"
    lite = f"rgb({min(r+14,255)},{min(g+12,255)},{min(b+10,255)})"

    dx = np.gradient(x)
    dy = np.gradient(y)
    nrm = np.hypot(dx, dy)
    nrm[nrm < 1e-9] = 1.0
    nx = dy / nrm
    ny = -dx / nrm

    h = layer_height * overlap
    z_bot = np.maximum(z - h, 0.0)
    z_mid = np.maximum(z - h / 2.0, 0.0)
    half = bead_width / 2.0

    vx = np.empty(4 * n); vy = np.empty(4 * n); vz = np.empty(4 * n)
    vx[0::4] = x;             vy[0::4] = y;             vz[0::4] = z_bot  # B
    vx[1::4] = x + nx * half; vy[1::4] = y + ny * half; vz[1::4] = z_mid  # O
    vx[2::4] = x;             vy[2::4] = y;             vz[2::4] = z      # T
    vx[3::4] = x - nx * half; vy[3::4] = y - ny * half; vz[3::4] = z_mid  # I
    vc = np.empty(4 * n, dtype=object)
    vc[0::4] = dark; vc[1::4] = mid; vc[2::4] = lite; vc[3::4] = mid

    ar = 4 * np.arange(n - 1, dtype=np.int32)
    Ii = np.empty((8, n - 1), np.int32)
    Jj = np.empty_like(Ii)
    Kk = np.empty_like(Ii)
    row = 0
    for a, bgn in ((0, 1), (1, 2), (2, 3), (3, 0)):   # B-O, O-T, T-I, I-B
        a0 = ar + a; a1 = a0 + 4
        b0 = ar + bgn; b1 = b0 + 4
        Ii[row] = a0; Jj[row] = a1; Kk[row] = b0; row += 1
        Ii[row] = b0; Jj[row] = a1; Kk[row] = b1; row += 1
    return {"vx": vx, "vy": vy, "vz": vz, "vc": vc,
            "ti": Ii.T.ravel(), "tj": Jj.T.ravel(), "tk": Kk.T.ravel(), "n": n}


def partial_bead_mesh(arrays, upto=None, name="Clay Body"):
    """Mesh3d showing the first `upto` path points of a bead_mesh_arrays build."""
    n = arrays["n"]
    p = n if upto is None else max(0, min(int(upto), n))
    if p < 2:
        return None
    t = 8 * (p - 1)
    return go.Mesh3d(
        x=arrays["vx"][:4 * p], y=arrays["vy"][:4 * p], z=arrays["vz"][:4 * p],
        i=arrays["ti"][:t], j=arrays["tj"][:t], k=arrays["tk"][:t],
        vertexcolor=arrays["vc"][:4 * p].tolist(),
        lighting=dict(ambient=0.42, diffuse=0.75, specular=0.18,
                      roughness=0.55, fresnel=0.05),
        lightposition=dict(x=200, y=100, z=400),
        flatshading=False,
        name=name, showlegend=False,
        hoverinfo="skip",
    )


def split_gapped(xs, ys, zs):
    """Turn None-separated coordinate lists into a list of (x, y, z) runs."""
    runs, cx, cy, cz = [], [], [], []
    for x, y, z in zip(xs, ys, zs):
        if x is None:
            if len(cx) >= 2:
                runs.append((cx, cy, cz))
            cx, cy, cz = [], [], []
        else:
            cx.append(x); cy.append(y); cz.append(z)
    if len(cx) >= 2:
        runs.append((cx, cy, cz))
    return runs


def toolpath_figure(groups, layer_height, bed_x, bed_y, height=560, bead_width=3.0):
    """
    Shaded preview figure.

    groups: list of (name, runs, hex_color, show_in_legend).
    """
    fig = go.Figure()
    # Bed: a subtle floor plate
    fig.add_trace(go.Mesh3d(
        x=[0, bed_x, bed_x, 0], y=[0, 0, bed_y, bed_y], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="rgb(235,232,228)", opacity=0.45,
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, bed_x, bed_x, 0, 0], y=[0, 0, bed_y, bed_y, 0], z=[0.05] * 5,
        mode="lines", line=dict(color="rgba(150,150,150,0.6)", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    for name, runs, color, legend in groups:
        mesh = runs_to_mesh(runs, layer_height, color, name=name, legend=legend,
                            bead_width=bead_width)
        if mesh is not None:
            fig.add_trace(mesh)
    fig.update_layout(
        uirevision="toolpath",   # keep the user's camera across reruns
        scene=dict(
            aspectmode="data",
            xaxis=dict(range=[0, bed_x], title="", showbackground=False,
                       gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)"),
            yaxis=dict(range=[0, bed_y], title="", showbackground=False,
                       gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)"),
            zaxis=dict(title="Z (mm)", showbackground=False,
                       gridcolor="rgba(0,0,0,0.08)"),
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
        height=height, margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(y=0.95, x=0.02),
    )
    return fig
