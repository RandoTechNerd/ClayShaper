import numpy as np
import math

class ClayObj:
    def __init__(self):
        self.path = []
        self.layer_height = 0.0
        self.nozzle_diameter = 1.0
        # Indices i where the move INTO path[i] is a travel (lift, no extrusion).
        # Used by multi-copy handles; empty for vessels (fully continuous).
        self.travel_idx = set()

    def add_point(self, x, y, z):
        self.path.append([x, y, z])

    def to_numpy(self):
        return np.array(self.path)

def generate_spiral_path(
    body_base_radius=50.0,
    total_height=100.0,
    layer_height=2.0,
    nozzle_diameter=2.0,

    base_height=0.0,
    base_bottom_radius=None,

    sides=60,
    profile_func=None,
    texture_func=None,
    first_layer_height=None,
):
    clay = ClayObj()
    clay.layer_height = layer_height
    clay.first_layer_height = first_layer_height or layer_height
    clay.nozzle_diameter = nozzle_diameter

    # Determine the target radius for the top of the base (start of the wall)
    # This ensures the base 'autofills' correctly into the wall shape.
    base_top_radius = body_base_radius
    if profile_func:
        base_top_radius = profile_func(0.0, body_base_radius)

    # Default the base footprint to the vessel's actual bottom radius so the
    # solid base is never wider than the wall it supports (important for
    # Virtual Wheel profiles whose bottom is narrower than the max radius).
    if base_bottom_radius is None:
        base_bottom_radius = base_top_radius

    # --- 1. BASE GENERATION (Alternating Spiral Autofill) ---
    num_base_layers = int(base_height / layer_height)
    if num_base_layers > 0:
        for layer in range(num_base_layers):
            curr_z = (layer + 1) * layer_height
            
            # Interpolate radius from base_bottom_radius to base_top_radius
            t_layer_norm = layer / (num_base_layers - 1) if num_base_layers > 1 else 1.0
            current_layer_outer_radius = base_bottom_radius + (base_top_radius - base_bottom_radius) * t_layer_norm
            
            # If radius is effectively zero, just add a center point
            if current_layer_outer_radius < nozzle_diameter / 2:
                clay.add_point(0, 0, curr_z)
                continue

            loops = int(current_layer_outer_radius / (nozzle_diameter * 0.95))
            loops = max(1, loops)
            points_per_spiral = max(loops * sides, sides) 
            
            # Logic: We want the LAST base layer to always be outward (In-to-Out)
            # so it meets the wall at the edge.
            # If total base layers is N, the last layer index is N-1.
            # We want (N-1) to be 'outward'.
            # Let's define: is_outward = (layer % 2 == (num_base_layers - 1) % 2)
            # This way, when layer == num_base_layers - 1, the condition is always True.
            
            is_outward = (layer % 2 == (num_base_layers - 1) % 2)
            
            for i in range(points_per_spiral + 1):
                t_spiral_norm = i / points_per_spiral 
                
                if is_outward:
                    # Center to edge
                    r = t_spiral_norm * current_layer_outer_radius 
                    angle = t_spiral_norm * loops * 2 * math.pi
                else:
                    # Edge to center
                    r = (1.0 - t_spiral_norm) * current_layer_outer_radius
                    angle = (1.0 - t_spiral_norm) * loops * 2 * math.pi
                
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                clay.add_point(x, y, curr_z)

    # --- 2. WALL GENERATION (Vase Mode) ---
    start_z = num_base_layers * layer_height
    wall_height = total_height - start_z
    
    if wall_height > 0:
        layers = int(wall_height / layer_height)
        total_points = layers * sides
        
        for i in range(total_points):
            t_wall_norm = i / total_points 
            current_z = start_z + t_wall_norm * wall_height
            angle = (i / sides) * 2 * math.pi
            
            # Base Radius from Profile
            r_base = body_base_radius
            if profile_func:
                # Pass body_base_radius to profile function
                r_base = profile_func(t_wall_norm, body_base_radius)
                
            r_offset = 0.0
            if texture_func:
                r_offset = texture_func(t_wall_norm, angle)
                
            r_final = r_base + r_offset
            x = r_final * math.cos(angle)
            y = r_final * math.sin(angle)
            clay.add_point(x, y, current_z)

    # First-layer height: shifting the whole path by (flh - lh) puts layer 1's
    # top at flh while every later layer keeps its normal spacing.
    dz = clay.first_layer_height - layer_height
    if abs(dz) > 1e-9:
        for p in clay.path:
            p[2] += dz

    return clay

# --- HANDLES (printed flat on the bed, attached to the pot after drying) ----
# The handle's curve lies in the bed plane; the strap's width is built up in Z.
# 1-bead thickness prints as a zigzag (forward one layer, back the next);
# 2-bead prints out along the inner offset and back along the outer — both are
# fully continuous. Multiple copies are separated by real (non-extruding)
# travel moves.

def _densify_poly(points_list, n=80):
    """Resample a polyline to n points evenly spaced by arc length."""
    pts = np.asarray(points_list, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])])


def _handle_curve(style, n=90, kuksa_d=0.985):
    """Unit open curve for a handle: bulge in +Y, attachment ends toward y=0.
    Scaled to real millimetres by _scale_curve afterwards.

    kuksa_d: lobe-center separation for the Kuksa (0.55 shallow dip … 0.985
    valley touches the attachment line, i.e. two circles meeting at the pot)."""
    if style in ("Half Circle", "Half Oval"):
        th = np.linspace(np.pi, 0.0, n)
        return np.column_stack([0.5 * np.cos(th), np.sin(th)])
    if style == "3/4 Circle":
        th = np.deg2rad(np.linspace(225.0, -45.0, n))   # over the top
        return np.column_stack([np.cos(th), np.sin(th)])
    if style == "Kuksa (Double)":
        # Two near-full circles side by side. Unit circles centered at ±d
        # intersect at (0, sqrt(1-d^2)), i.e. at angle acos(d) from each
        # center — sweeping each lobe to exactly that angle joins them with
        # no gap for ANY d. As d -> 1 the valley drops to the attachment
        # line: two separate circles touching at the pot (classic kuksa).
        # Ends sweep to 182deg/-2deg so they sit ON the attachment line and
        # the valley can genuinely reach it.
        d = float(np.clip(kuksa_d, 0.3, 0.9995))
        va = np.degrees(np.arccos(d))
        half = n // 2
        a1 = np.deg2rad(np.linspace(182.0, va, half))
        a2 = np.deg2rad(np.linspace(180.0 - va, -2.0, n - half))
        left = np.column_stack([-d + np.cos(a1), np.sin(a1)])
        right = np.column_stack([d + np.cos(a2), np.sin(a2)])
        return np.vstack([left, right])
    if style in ("Half Square", "Half Rectangle"):
        return _densify_poly([(-0.5, 0), (-0.5, 1), (0.5, 1), (0.5, 0)], n)
    if style == "Half Triangle":
        return _densify_poly([(-0.5, 0), (0.0, 1), (0.5, 0)], n)
    if style == "Hook (7)":
        # Classic open-bottom handle: out from the rim, then straight down.
        return _densify_poly([(-0.5, 1), (0.5, 1), (0.5, 0)], n)
    if style == "Half Heart":
        # One side of the classic parametric heart; both ends land on the
        # attachment line, the lobe bulges outward.
        t = np.linspace(0.0, np.pi, n)
        hx = 16 * np.sin(t) ** 3
        hy = (13 * np.cos(t) - 5 * np.cos(2 * t)
              - 2 * np.cos(3 * t) - np.cos(4 * t))
        return np.column_stack([-hy, hx])   # rotate: attachment line -> y=0
    raise ValueError(f"Unknown handle style: {style}")


def _scale_curve(pts, width, height):
    """Center x and rescale so the bounding box is exactly width x height."""
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    x -= (x.min() + x.max()) / 2.0
    xspan = max(x.max() - x.min(), 1e-9)
    y -= y.min()
    yspan = max(y.max(), 1e-9)
    return np.column_stack([x * (width / xspan), y * (height / yspan)])


HANDLE_STYLES = ["Half Circle", "Half Oval", "3/4 Circle", "Kuksa (Double)",
                 "Half Square", "Half Rectangle", "Half Triangle",
                 "Half Heart", "Hook (7)"]


def generate_handle_path(style="Half Circle", width=60.0, height=35.0,
                         strap_width=16.0, layer_height=1.0,
                         nozzle_diameter=3.0, thickness_beads=1,
                         copies=1, spacing=None, first_layer_height=None,
                         kuksa_d=0.985):
    """
    Flat-printed handle(s) for cups/mugs: the curve lies on the bed and the
    strap width is built up as print height. Once leather-hard, the user
    scores + slips it onto the vessel.

    thickness_beads (1-6): parallel offset passes traced as a serpentine
    within each layer; consecutive layers retrace it in reverse, so the whole
    handle is one continuous bead regardless of thickness.
    copies: printed side by side, separated by real travel moves.
    kuksa_d: Kuksa center-dip control (see _handle_curve).
    """
    clay = ClayObj()
    clay.layer_height = layer_height
    clay.first_layer_height = first_layer_height or layer_height
    clay.nozzle_diameter = nozzle_diameter

    curve = _scale_curve(_handle_curve(style, kuksa_d=kuksa_d), width, height)
    layers = max(2, int(round(strap_width / layer_height)))
    flh = clay.first_layer_height
    if spacing is None:
        spacing = width + 18.0
    copies = max(1, int(copies))
    beads = max(1, min(int(thickness_beads), 6))

    # One layer = a serpentine over `beads` parallel offset curves (centered
    # on the drawn line). The next layer retraces it exactly backwards, so
    # layer ends always meet: continuous for any bead count.
    # Offsets use shapely's geometric offsetting — naive per-point normals
    # spike wildly at sharp corners (squares, deep kuksa valleys) once the
    # offset exceeds the local curvature radius.
    if beads == 1:
        layer_pts = curve
    else:
        from shapely.geometry import LineString, MultiLineString
        base_ls = LineString(curve)
        passes = []
        for k in range(beads):
            off = (k - (beads - 1) / 2.0) * nozzle_diameter
            if abs(off) < 1e-9:
                c = curve
            else:
                oc = base_ls.offset_curve(off, join_style="round")
                if isinstance(oc, MultiLineString):
                    oc = max(oc.geoms, key=lambda g: g.length)   # keep the main run
                c = np.asarray(oc.coords)
                if len(c) < 2:
                    c = curve
                # offset curves can come back with reversed orientation;
                # align each pass to the drawn direction before serpentining.
                if (np.linalg.norm(c[0] - curve[0]) >
                        np.linalg.norm(c[-1] - curve[0])):
                    c = c[::-1]
                c = _densify_poly(c, max(len(curve), 60))
            passes.append(c if k % 2 == 0 else c[::-1])
        layer_pts = np.vstack(passes)

    for ci in range(copies):
        x0 = (ci - (copies - 1) / 2.0) * spacing
        if ci > 0:
            clay.travel_idx.add(len(clay.path))   # hop to the next copy
        for L in range(layers):
            z = flh + L * layer_height
            pts = layer_pts if L % 2 == 0 else layer_pts[::-1]
            for px, py in pts:
                clay.add_point(px + x0, py, z)
    return clay


# --- HELPERS ---
# Wall profiles: r(t) for t = 0 at the foot to t = 1 at the rim.

def profile_cylinder(t, r_base, taper=1.0):
    """Straight wall, optionally tapered (taper is the rim vs foot ratio)."""
    return r_base * (1.0 + (taper - 1.0) * t)


def profile_bowl(t, r_base, flare=1.0, belly=0.45):
    """A bowl, from its foot (t=0) to its rim (t=1).

    Two independent controls, because one was not enough to describe a bowl:
      flare  rim radius as a fraction of the foot. 1.0 keeps the rim the same
             width as the base; above 1.0 opens out.
      belly  how far the middle bows past that line, which is what makes a
             bowl look thrown rather than machined.

    With flare=1.0 the result is a rounded barrel: base and rim equal, widest
    across the middle. A linear r_base * (1 + t), the original formula, was a
    straight sided cone and could not produce this shape at any setting.
    """
    ease = math.sin(t * math.pi / 2.0)          # quick off the foot, easing to the rim
    return r_base * ((1.0 - ease) + flare * ease + belly * math.sin(math.pi * t))


def profile_vase(t, r_base, belly=0.45, neck=0.65):
    """A vase: swells at the belly, then draws back in to a narrower neck.

    `neck` is the rim radius as a fraction of the foot. The old version added
    a sine bulge only, and since sin(pi) is 0 the rim came back to the full
    foot radius, so it never actually necked in.
    """
    taper = (1.0 - t) + neck * t                     # foot -> neck
    return r_base * (taper + belly * math.sin(math.pi * t))

def texture_sine_waves(t, angle, freq=10, amp=2.0): return amp * math.sin(angle * freq)
def texture_twist(t, angle, twist_factor=2.0, ridges=5, amp=3.0):
    phase = angle + t * twist_factor * 2 * math.pi
    return amp * math.sin(phase * ridges)

def texture_from_image(t, angle, img_array, amp=5.0):
    """
    Maps an image onto the cylinder surface.
    img_array: 2D numpy array (normalized 0.0 to 1.0), shape (H, W)
    t: normalized height (0.0 to 1.0)
    angle: rotation angle (radians)
    """
    if img_array is None:
        return 0.0
        
    h, w = img_array.shape
    
    # Map t (0..1) to y pixel coordinate (0..h-1)
    # Note: t=0 is bottom, usually images are y=0 at top. 
    # Let's flip y so image looks upright on the pot.
    py = int((1.0 - t) * h)
    
    # Map angle to x pixel coordinate. 
    # Angle grows continuously. Normalize to 0..1 per revolution.
    # angle / (2*pi)
    norm_angle = (angle / (2 * math.pi)) % 1.0
    px = int(norm_angle * w)
    
    # Clamp to ensure indices are valid
    py = max(0, min(py, h - 1))
    px = max(0, min(px, w - 1))
    
    val = img_array[py, px]
    return val * amp

def generate_stl_from_path(clay_obj, output_filename):
    """
    Simple function to generate an STL tube from the path points.
    We'll treat each segment as a simple quad-strip or just save the raw points as a tube.
    For simplicity/robustness without heavy libs like trimesh, we'll manually write binary STL 
    or just use numpy-stl if available?
    The user environment seems to have numpy-stl (from Possum project).
    """
    # Let's check imports
    try:
        from stl import mesh
    except ImportError:
        return False
        
    # float64 throughout: integer coordinates (e.g. center points at 0,0) would
    # otherwise make np.cross return int arrays and `v1 /= norm` blow up.
    points = clay_obj.to_numpy().astype(np.float64)
    if len(points) < 2:
        return False

    # Create a simple tube mesh
    # We will generate quads between point i and i+1
    # But wait, the points are a single line. A 3D printer path is 1D.
    # An STL needs volume. We can approximate the "clay bead" as a square or hex tube.
    
    # Simple approach: Create a tube with `sides` segments around each point
    
    # Collect all vertices and faces
    vertices = []
    faces = []
    
    # We need a tube radius -> nozzle / 2
    r = clay_obj.nozzle_diameter / 2.0
    tube_sides = 8
    
    for i in range(len(points)):
        p = points[i]
        # Create a ring of vertices at this point
        # Tangent estimation
        if i < len(points) - 1:
            tangent = points[i+1] - p
        else:
            tangent = p - points[i-1]
            
        # Normalize tangent
        norm = np.linalg.norm(tangent)
        if norm > 0: tangent /= norm
        else: tangent = np.array([0, 0, 1])
        
        # Calculate normal vectors to the tangent (basis for the ring)
        # Random vector not parallel to tangent
        if abs(tangent[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])
            
        v1 = np.cross(tangent, arbitrary)
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 1e-12:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = v1 / v1_norm
        v2 = np.cross(tangent, v1)
        
        # Create ring vertices
        start_idx = len(vertices)
        for j in range(tube_sides):
            theta = j * 2 * math.pi / tube_sides
            offset = r * (math.cos(theta) * v1 + math.sin(theta) * v2)
            vertices.append(p + offset)
            
        # Create faces connecting to previous ring
        if i > 0:
            prev_ring_start = start_idx - tube_sides
            curr_ring_start = start_idx
            
            for j in range(tube_sides):
                next_j = (j + 1) % tube_sides
                
                # Quad = 2 triangles
                # v0 = prev[j], v1 = prev[next], v2 = curr[next], v3 = curr[j]
                
                p0 = prev_ring_start + j
                p1 = prev_ring_start + next_j
                p2 = curr_ring_start + next_j
                p3 = curr_ring_start + j
                
                faces.append([p0, p1, p2])
                faces.append([p0, p2, p3])
                
    # Create the mesh object
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    clay_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            clay_mesh.vectors[i][j] = vertices[f[j], :]
            
    clay_mesh.save(output_filename)
    return True

# -----------------------------------------------------------------------------
# Printer profiles
# -----------------------------------------------------------------------------
# The Eazao clay printers run a Marlin fork with a dual-material *mixing* hotend.
# Two things are mandatory or the machine will NOT extrude clay:
#   * M302  - allow "cold" extrusion (there is no heater; without this the
#             firmware blocks all E moves).
#   * M163/M164 - set the mixing ratio for the two virtual extruders.
# The start/end blocks below are lifted verbatim from Eazao's own Cura output
# (see Dev/Eazao Potter/Bowl.gcode) so generated files behave identically.

EAZAO_START_GCODE = """M105
M109 S0
M82 ;absolute extrusion mode
G21
G90 ;absolute positioning
M82 ;set extruder to absolute mode
G28 ;Home
G1 Z25.0 F1500 ;move the platform down 25mm
G4 S3; wait 3 seconds
G92 E0
G1 F1500 E2
G92 E0

M302 ;allow cold extrusion (clay has no heater)
M163 S0 P0.85; Set Mix Factor
M163 S1 P0.15; Set Mix Factor
M164 S0
G92 E0"""

EAZAO_END_GCODE = """M107
G92 Z0 E0
G1 F1500 Z10 E-2 ;lift platform and relieve pressure
M82
M84 ;steppers off
;End of Gcode"""

# Machine limits below are from the official manuals / factory G-code, not guesses:
#   Potter — "20250812-Eazao Potter Manual.pdf" p.5: build volume 165x165x280,
#            nozzle 1.6-3.3mm, print speed 10-40 mm/s (=600-2400 mm/min).
#   max_feedrate is a mechanical hard cap (travel / Z-hop moves in the factory
#   files reach ~3300 mm/min), used only to catch truly runaway feedrates.
#   cartridge_ml is the clay reservoir volume, used for the "too much material" check.
PRINTER_PROFILES = {
    "Eazao Potter": {
        "bed_x": 165.0, "bed_y": 165.0, "max_z": 280.0,
        "center_x": 82.5, "center_y": 82.5,
        "print_speed": 1500,      # F, mm/min  (25 mm/s) - default extrude speed
        "z_speed": 300,           # F, mm/min for pure-Z approach moves
        "min_print_speed": 600,   # 10 mm/s
        "max_print_speed": 2400,  # 40 mm/s  (spec max for extruding moves)
        "max_feedrate": 3600,     # mechanical hard cap (any move)
        "layer_range": (0.4, 1.0),
        "cartridge_ml": 500.0,
        "filament_dia": 1.75,     # E is volumetric against this area (matches Eazao Cura)
        "nozzles": [1.6, 2.0, 3.0, 3.3],
        "start_gcode": EAZAO_START_GCODE,
        "end_gcode": EAZAO_END_GCODE,
    },
    "Eazao Zero": {
        "bed_x": 160.0, "bed_y": 160.0, "max_z": 150.0,
        "center_x": 80.0, "center_y": 80.0,
        "print_speed": 1500,
        "z_speed": 300,
        "min_print_speed": 600,
        "max_print_speed": 2400,
        "max_feedrate": 3600,
        "layer_range": (0.4, 1.0),
        "cartridge_ml": 500.0,
        "filament_dia": 1.75,
        "nozzles": [1.6, 2.0, 3.0],
        "start_gcode": EAZAO_START_GCODE,
        "end_gcode": EAZAO_END_GCODE,
    },
}


def generate_gcode(clay_obj, offset_x=0.0, offset_y=0.0, profile=None, line_width=None,
                   first_layer_flow=1.0, source=None):
    """
    Generate Eazao-compatible G-code for a clay path.

    profile: a dict from PRINTER_PROFILES (defaults to Eazao Potter). Controls the
             mandatory start/end blocks, feedrates and the volumetric E scale.
    line_width: extruded bead width used for the volume calc. Defaults to the
                object's nozzle_diameter (Eazao runs line width == nozzle size).
    first_layer_flow: extrusion multiplier applied to the first layer only
                      (e.g. 1.2 = 20% extra clay for bed adhesion).
    source: human-readable provenance written as a ;SOURCE: header comment.
    """
    if profile is None:
        profile = PRINTER_PROFILES["Eazao Potter"]
    if line_width is None:
        line_width = clay_obj.nozzle_diameter

    points = clay_obj.to_numpy()
    if len(points) == 0:
        return ""

    # Volumetric extrusion: E advances by (bead volume / filament cross-section).
    # bead volume per mm of travel = layer_height * line_width.
    # This reproduces Eazao's own numbers (~1.247 E/mm at 1mm x 3mm) exactly.
    filament_area = math.pi * (profile["filament_dia"] / 2.0) ** 2
    f_print = profile["print_speed"]
    f_z = profile["z_speed"]

    # Toolpath bounds (for the informational header Cura-style comments).
    xs = points[:, 0] + offset_x
    ys = points[:, 1] + offset_y
    zs = points[:, 2]

    g = []
    g.append(";FLAVOR:Marlin")
    g.append(";Generated by ClayShaper")
    if source:
        g.append(f";SOURCE: {source}")
    if first_layer_flow != 1.0:
        g.append(f";First layer flow: {first_layer_flow*100:.0f}%")
    g.append(f";Layer height: {clay_obj.layer_height:g}")
    g.append(f";Line width: {line_width:g}")
    g.append(f";MINX:{xs.min():.2f}")
    g.append(f";MINY:{ys.min():.2f}")
    g.append(f";MINZ:{zs.min():.2f}")
    g.append(f";MAXX:{xs.max():.2f}")
    g.append(f";MAXY:{ys.max():.2f}")
    g.append(f";MAXZ:{zs.max():.2f}")
    g.append(profile["start_gcode"])

    total_e = 0.0

    # Approach the first point: travel in XY at safe height, then drop to Z.
    start_pt = points[0]
    sx, sy, sz = start_pt[0] + offset_x, start_pt[1] + offset_y, start_pt[2]
    g.append("M107")
    g.append(f"G0 F{f_print} X{sx:.3f} Y{sy:.3f} Z{sz + 2:.3f}")
    g.append(f"G1 F{f_z} Z{sz:.3f}")
    g.append(f"G1 F{f_print} E0")

    flh = getattr(clay_obj, "first_layer_height", clay_obj.layer_height)
    first_layer_top = flh + 1e-6
    travels = getattr(clay_obj, "travel_idx", None) or set()
    for i in range(1, len(points)):
        p_prev = points[i - 1]
        p_curr = points[i]

        if i in travels:
            # Non-extruding hop (e.g. between handle copies): lift, move, drop.
            cx, cy, cz = p_curr[0] + offset_x, p_curr[1] + offset_y, p_curr[2]
            g.append(f"G0 F{f_print} Z{p_prev[2] + 5:.3f}")
            g.append(f"G0 F{f_print} X{cx:.3f} Y{cy:.3f} Z{cz + 2:.3f}")
            g.append(f"G1 F{f_z} Z{cz:.3f}")
            continue

        dist = float(np.linalg.norm(p_curr - p_prev))

        on_first = p_curr[2] <= first_layer_top
        bead_h = flh if on_first else clay_obj.layer_height
        flow = first_layer_flow if on_first else 1.0
        total_e += (dist * bead_h * line_width * flow) / filament_area

        cx, cy, cz = p_curr[0] + offset_x, p_curr[1] + offset_y, p_curr[2]
        g.append(f"G1 X{cx:.3f} Y{cy:.3f} Z{cz:.3f} E{total_e:.5f}")

    g.append(profile["end_gcode"])
    return "\n".join(g)