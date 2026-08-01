"""
Generate ClayShaper's own sample models.

The models we developed against belong to Eazao and other designers, so they
can't ship with a public build. These four are generated from scratch here —
original work, ours to distribute — and they're deliberately chosen to show
what the slicer does well: a wide bowl, a bellied vase, a rippled cup and a
twisted pot.

Run:  python tools/make_samples.py
"""
import math
import os

import numpy as np
import trimesh

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "assets", "models")
SIDES = 128          # smooth enough for a 3 mm nozzle, still a small file
RINGS = 120


def vessel(profile, height, texture=None, sides=SIDES, rings=RINGS):
    """Solid of revolution from a radius profile r(t), t in 0..1 (bottom->top).

    texture(t, angle) may add a small radial offset for ripples/twists.
    Returns a watertight trimesh with a flat base and a closed top.
    """
    verts, faces = [], []
    for i in range(rings + 1):
        t = i / rings
        z = t * height
        r_base = profile(t)
        for j in range(sides):
            a = 2 * math.pi * j / sides
            r = r_base + (texture(t, a) if texture else 0.0)
            r = max(r, 0.05)
            verts.append((r * math.cos(a), r * math.sin(a), z))
    n_side = (rings + 1) * sides
    for i in range(rings):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + j
            d = (i + 1) * sides + (j + 1) % sides
            faces += [(a, c, d), (a, d, b)]
    # Caps: a centre vertex fanned to the first/last ring.
    bot_c, top_c = n_side, n_side + 1
    verts += [(0.0, 0.0, 0.0), (0.0, 0.0, height)]
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((bot_c, j2, j))                                  # base
        faces.append((top_c, rings * sides + j, rings * sides + j2))  # top
    m = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64),
                        faces=np.array(faces, dtype=np.int64), process=True)
    m.fix_normals()
    return m


def save(mesh, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".stl")
    mesh.export(path)
    kb = os.path.getsize(path) / 1024
    print(f"  {name:16s} {len(mesh.faces):6d} faces  {kb:6.0f} KB  "
          f"watertight={mesh.is_watertight}  "
          f"size={np.round(mesh.extents, 1).tolist()}")


print("Generating ClayShaper sample models…")

# 1. Coil Bowl — wide, gently flared: the classic first print.
save(vessel(lambda t: 26 + 34 * math.sin(t * math.pi / 2), height=62), "Coil Bowl")

# 2. Belly Vase — narrow foot, full belly, drawn-in neck.
save(vessel(lambda t: 20 + 26 * math.sin(math.pi * min(1.0, t * 0.92 + 0.06)),
            height=150), "Belly Vase")

# 3. Ripple Cup — vertical fluting; shows texture following the wall.
save(vessel(lambda t: 32 + 5 * t, height=95,
            texture=lambda t, a: 2.2 * math.sin(a * 14)), "Ripple Cup")

# 4. Twist Pot — ridges that rotate as they rise.
save(vessel(lambda t: 30 + 12 * math.sin(t * math.pi), height=120,
            texture=lambda t, a: 2.6 * math.sin(6 * (a + t * 2.4 * math.pi))),
     "Twist Pot")

print("Done — models written to assets/models/")
