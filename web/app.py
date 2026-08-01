"""ClayShaper — client-side web build (stlite/Pyodide). Same UI as the desktop
app; only the pieces stlite can't run are swapped: the Virtual Wheel uses a
plain HTML5 canvas (postMessage -> controller.writeFile bridge in index.html)
instead of the drawable-canvas custom component, and gallery/printer selection
uses native widgets instead of ?query URL navigation (a navigation would
remount the whole Python runtime)."""
import streamlit as st
import numpy as np
from PIL import Image

# --- WASM shim ---------------------------------------------------------------
# Pyodide's numpy is 32-bit (np.intp == int32); trimesh calls np.bincount on
# int64 edge arrays and numpy refuses the "safe" downcast that 64-bit native
# builds never hit. Mesh index counts are tiny, so the cast is lossless.
if np.intp().itemsize < 8:
    _orig_bincount = np.bincount
    def _safe_bincount(x, *a, **k):
        x = np.asarray(x)
        if x.dtype.kind in "iu" and x.dtype.itemsize > np.intp().itemsize:
            x = x.astype(np.intp)
        return _orig_bincount(x, *a, **k)
    np.bincount = _safe_bincount

import streamlit.components.v1 as components
import plotly.graph_objects as go
from clay_lib import generate_spiral_path, profile_cylinder, profile_bowl, profile_vase, texture_sine_waves, texture_twist, texture_from_image, generate_stl_from_path, generate_gcode, PRINTER_PROFILES, EAZAO_START_GCODE, EAZAO_END_GCODE, generate_handle_path, HANDLE_STYLES
from gcode_validator import validate_gcode, extract_toolpath, FAIL, WARN, SUGGEST, INFO
from stl_slicer import slice_stl
from sample_library import list_sample_stls, list_sample_gcodes, get_thumbnail, get_thumbnail_b64
from preview3d import split_gapped, toolpath_figure, bead_mesh_arrays, partial_bead_mesh
import brand
from brand import PRINTER_CATALOG, printer_icon, wordmark_svg, ui_icon, THEMES
import tempfile
import os
import re
import json
from urllib.parse import quote as _q
import time
import hashlib
from datetime import datetime

# Resolve asset paths relative to this file so the app runs from any CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(HERE, "clayshaper_icon.png")
EAZAO_DIR = os.path.normpath(os.path.join(HERE, "..", "Eazao Potter"))

# External links.
LINK_RTN = "https://www.youtube.com/@RandoTechNerd"      # YouTube channel
LINK_RTN_SITE = "https://randotechnerd.com"             # personal site / other projects
CONTACT_EMAIL = "RandoTechNerd@gmail.com"
# Plus + donations both go through Buy Me a Coffee for now: supporters who
# donate get set up with the local desktop version.
LINK_PLUS = "https://buymeacoffee.com/randotechnerd"
LINK_DONATE = "https://buymeacoffee.com/randotechnerd"
LINK_EAZAO = "https://www.eazao.com"
LINK_HOME = "https://clayshaper.com"

st.set_page_config(page_title="ClayShaper", page_icon=ICON_PATH, layout="wide")

# --- URL HANDLERS (must run before any widget is created) ---
# Gallery cards navigate to ?sample=<name>; the printer dropdown to ?printer=<name>.
# Both start a fresh script run, so catch them here, update state, clean the URL.
st.session_state.setdefault("printer_name", "Eazao Potter")

# User-added printers (BETA), created via "+ Add your printer". Kept on disk —
# dropdown links navigate (new Streamlit session), so session_state alone
# would silently drop them.
_CUSTOM_PRINTERS_FILE = os.path.join(HERE, "custom_printers.json")


def _load_custom_printers():
    try:
        with open(_CUSTOM_PRINTERS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        for p in d.values():
            p["layer_range"] = tuple(p.get("layer_range", (0.4, 1.0)))
        return d
    except Exception:
        return {}


def _save_custom_printers(d):
    try:
        with open(_CUSTOM_PRINTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


st.session_state.setdefault("custom_printers", _load_custom_printers())
ALL_PROFILES = {**PRINTER_PROFILES, **st.session_state.custom_printers}
if st.session_state.printer_name not in ALL_PROFILES:
    st.session_state.printer_name = "Eazao Potter"

_picked_printer = st.query_params.get("printer")
_picked_sample = st.query_params.get("sample")
if _picked_printer or _picked_sample:
    if _picked_printer == "__add__":
        st.session_state.show_add_printer = True
    elif _picked_printer and _picked_printer in ALL_PROFILES:
        st.session_state.printer_name = _picked_printer
    if _picked_sample:
        _by_name = {name: path for name, _, path in list_sample_stls()}
        if _picked_sample in _by_name:
            # Loading a sample replaces whatever model was active.
            st.session_state.current_model = ("sample", _picked_sample, _by_name[_picked_sample])
            st.session_state.mode_tabs = "Slice STL"
    st.query_params.clear()

selected_printer = st.session_state.printer_name

# Theme (set in Advanced menu; widget state persists across runs, so reading it
# here — before the widget renders — still gets the current value). Themes
# restyle the whole app: accent, sidebar, background and text.
_theme = THEMES.get(st.session_state.get("theme_accent", "Terracotta"), THEMES["Terracotta"])
ACC, ACC_DK = _theme["acc"], _theme["acc_dk"]
INKC, MUTEDC = _theme["ink"], _theme["muted"]
SIDE_BG, APP_BG, LINEC = _theme["side"], _theme["app"], _theme["line"]

# User-adjustable preview clay color + material units (set in Advanced).
CLAY = st.session_state.get("clay_color", "#B0603C")
BASE_CLR = st.session_state.get("base_color", "#3D6EA8")
MAT_UNIT = st.session_state.get("mat_unit", "grams")


def _lighten(hexcolor, amt=0.32):
    """Blend a hex color toward white by amt (0..1) — used for the alternating
    'base B' stagger tint so it's visibly lighter than 'base A'."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * amt)
    g = int(g + (255 - g) * amt)
    b = int(b + (255 - b) * amt)
    return f"#{r:02X}{g:02X}{b:02X}"

# Wet-clay density ~1.9 g/ml (stoneware). Everything downstream of ml uses this;
# it is an estimate only — real clay bodies and water content vary a lot.
_CLAY_G_PER_ML = 1.9


def format_material(ml, unit=None):
    unit = unit or MAT_UNIT
    g = ml * _CLAY_G_PER_ML
    return {
        "ml": f"{ml:.0f} ml",
        "grams": f"{g:.0f} g",
        "kilograms": f"{g/1000:.2f} kg",
        "pounds": f"{g/453.592:.2f} lb",
    }.get(unit, f"{g:.0f} g")


# --- HELPERS ---
def render_validation(text, profile, nozzle=None, layer_height=None):
    """Run the validator on a G-code string and render a pass/warn/fail report.
    Returns the report so callers can gate the download button."""
    rep = validate_gcode(text, profile, nozzle=nozzle, layer_height=layer_height)

    verdict = rep.verdict
    if verdict == "fail":
        st.error("FAIL — do not print this file. See the issues below.",
                 icon=":material/block:")
    elif verdict == "caution":
        st.warning("CAUTION — printable, but review the warnings below first.",
                   icon=":material/warning:")
    elif verdict == "pass_suggest":
        st.info("PASS — ready to print, with suggestions to make it better.",
                icon=":material/lightbulb:")
    else:
        st.success("PASS — all checks clean. Ready to print.",
                   icon=":material/check_circle:")

    s = rep.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Footprint (mm)", f"{s['min_x']}–{s['max_x']} X")
    m2.metric("Height (mm)", s["max_z"] if s["max_z"] is not None else "—")
    m3.metric("Layers", s["layers"])
    m4.metric("Clay needed ~", format_material(s["clay_ml"]),
              help="Estimate only — actual clay use varies with clay body, "
                   "water content and how the machine is calibrated.")

    dot_colors = {FAIL: "#B3402A", WARN: "#D08A2E", SUGGEST: "#3D6EA8", INFO: "#5E8B6F"}
    for issue in rep.issues:
        line = f" · line {issue.line_no}" if issue.line_no else ""
        dot = (f"<span style='display:inline-block;width:10px;height:10px;"
               f"border-radius:50%;background:{dot_colors.get(issue.severity, '#999')};"
               f"margin-right:9px'></span>")
        st.markdown(f"{dot}**{issue.category}** — {issue.message}{line}",
                    unsafe_allow_html=True)
    return rep


def process_canvas_to_profile(img_data, height, width):
    """
    Turn a sketch into a radius profile: for each row, the right-most inked
    pixel is the outer wall. Returns a normalized array (0..1, bottom→top),
    or None if the canvas is effectively empty.

    Cleanup that makes hand drawings print sensibly:
      * trim empty rows above/below the ink — the LOWEST drawn point becomes the
        vessel bottom (its connection to the wheel base), the highest the rim;
      * interpolate across internal gaps (broken strokes);
      * light smoothing to remove pixel/antialiasing jitter.
    """
    h, w, _ = img_data.shape
    alpha = img_data[:, :, 3]
    prof = np.zeros(h)
    for y in range(h):
        ink = np.where(alpha[y] > 24)[0]
        if len(ink):
            prof[y] = ink.max() / w
    prof = prof[::-1]  # bottom -> top

    inked = np.nonzero(prof > 0)[0]
    if len(inked) == 0:
        return None
    prof = prof[inked[0]:inked[-1] + 1]          # trim to drawn extent

    zeros = prof == 0
    if zeros.any():                              # bridge stroke gaps
        idx = np.arange(len(prof))
        prof[zeros] = np.interp(idx[zeros], idx[~zeros], prof[~zeros])

    if len(prof) > 7:                            # gentle 5-tap smoothing
        kernel = np.ones(5) / 5.0
        prof = np.convolve(prof, kernel, mode="same")
        # convolve shrinks the ends toward 0; restore them
        prof[:2] = prof[2]
        prof[-2:] = prof[-3]
    return prof


def make_wheel_guide(w, h, trace_img=None):
    """Background image for the Virtual Wheel canvas: the center axis, the
    'wheel base' connection line, and a faint example silhouette so the intent
    is obvious at a glance. A user trace image replaces the example ghost."""
    from PIL import ImageDraw, ImageFont
    if trace_img is not None:
        img = trace_img.convert("RGB").resize((w, h))
    else:
        img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
        font_b = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 19)
    except OSError:
        font = font_b = ImageFont.load_default()

    axis_c = (176, 96, 60)        # terracotta
    base_c = (61, 74, 92)         # slate
    ghost_c = (208, 200, 190)

    # Center axis (left edge): dashed vertical line
    for y in range(0, h, 26):
        d.line([(10, y), (10, y + 14)], fill=axis_c, width=3)
    d.text((22, 12), "CENTER AXIS", fill=axis_c, font=font_b)
    d.text((22, 36), "the middle of your pot", fill=(170, 150, 140), font=font)

    # Wheel base (bottom edge): solid line + label
    d.line([(0, h - 26), (w, h - 26)], fill=base_c, width=3)
    d.text((22, h - 22), "WHEEL BASE — draw your wall down to this line so the pot connects to its foot",
           fill=base_c, font=font)

    if trace_img is None:
        # Ghost example: a vase silhouette from the base line up
        pts = []
        for i in range(61):
            t = i / 60
            y = (h - 26) - t * (h - 90)
            r = 0.16 + 0.14 * np.sin(t * np.pi) + 0.05 * np.sin(3 * t * np.pi)
            pts.append((r * w, y))
        for i in range(0, len(pts) - 1, 2):      # dashed
            d.line([pts[i], pts[i + 1]], fill=ghost_c, width=5)
        gx, gy = pts[len(pts) // 2]
        d.text((gx + 26, gy - 10), "e.g. — draw your pot's side wall like this, then press SPIN",
               fill=(185, 175, 165), font=font)
    return img


def model_thumb_b64(kind, name, payload):
    """Small sliced-preview thumbnail for the loaded model. Samples reuse the
    cached gallery thumbnail; uploads are rendered once and cached by content
    hash so the model row can show a little 'window' of what's loaded."""
    if kind == "sample":
        return get_thumbnail_b64(payload)
    h = hashlib.md5(payload).hexdigest()
    cache = st.session_state.setdefault("_upload_thumbs", {})
    if h in cache:
        return cache[h]
    b64 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp.write(payload)
            tp = tmp.name
        b64 = get_thumbnail_b64(tp)
        os.unlink(tp)
    except Exception:
        b64 = None
    cache[h] = b64
    return b64


def _doc_download(label, filename, icon=":material/description:"):
    """Offer a bundled Eazao document as a download if it's present on disk."""
    path = os.path.join(EAZAO_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(label, f.read(), file_name=filename, icon=icon,
                               use_container_width=True, key=f"doc_{filename}")
    else:
        # Deployed builds don't bundle Eazao's PDFs — link to the source.
        st.link_button(f"{label} — at eazao.com", LINK_EAZAO,
                       icon=":material/open_in_new:", use_container_width=True)


@st.dialog("Setup, Tips & Tricks", width="large")
def setup_tips_dialog():
    tab_setup, tab_clay, tab_video, tab_plus = st.tabs(
        ["Setup & Docs", "Clay Tips", "Videos", "ClayShaper Plus"])

    with tab_setup:
        st.markdown("#### Get the printer running")
        st.markdown(
            "1. **Assemble** the Potter and electric putter (part list + steps in the "
            "assembly tutorial below).\n"
            "2. **Level** the platform and install the clay barrel.\n"
            "3. **Pre-extrude** until clay flows evenly from the nozzle.\n"
            "4. **Slice** a model here → copy the `.gcode` to the SD card → print.")
        st.markdown("**Eazao documentation**")
        c1, c2 = st.columns(2)
        with c1:
            _doc_download("Potter Manual (PDF)", "20250812-Eazao Potter Manual.pdf")
            _doc_download("Assembly & Part List (PDF)",
                          "20250812-Eazao Potter 500ml Electric Putter Assembly Tutorial.pdf",
                          icon=":material/build:")
        with c2:
            _doc_download("Manual Instructions (PDF)", "Eazao's instructions for the manual.pdf")
            st.link_button("Eazao — parts & support", LINK_EAZAO,
                           icon=":material/open_in_new:", use_container_width=True)
        st.info("**Adding your own models?** ClayShaper ships with ready sample "
                "models, and uploading & slicing your own STLs is **always free** — "
                "just drop a file on the Slice STL tab. See the ClayShaper Plus tab "
                "if you'd like a local desktop copy or to support development.",
                icon=":material/info:")
        st.markdown("**Placeholder** — “EASY mode” setup + Cura/Orca walkthrough "
                    f"(coming to [RandoTechNerd]({LINK_RTN}))")

    with tab_clay:
        st.markdown("#### Choosing the right clay")
        st.markdown(
            "- Use a **smooth, plastic clay body** — stoneware or porcelain with little "
            "or no grog. Heavy grog clogs the 2–3 mm nozzle.\n"
            "- Avoid very short (crumbly) clays; you want it **plastic and cohesive**.")
        st.markdown("#### Mixing & wedging — get it *sticky*")
        st.markdown(
            "The single biggest factor in a clean print is water content. You want the "
            "clay **soft and sticky** — wetter than throwing clay:\n\n"
            "- **The slap test:** slap the ball firmly with an open hand. When you pull "
            "away, you should see **little spikes/peaks pulling up on your fingers**. "
            "That tackiness means it will bond layer-to-layer instead of tearing.\n"
            "- If it feels stiff or the surface cracks as you bend it, **wedge in more "
            "water** (spray + fold) until it slaps sticky.\n"
            "- Too wet (slumps, won't hold a coil) → wedge on a dry plaster bat to pull "
            "water back out.\n"
            "- **Wedge thoroughly** to remove air pockets — a trapped bubble = a blowout "
            "mid-print.")
        st.markdown("#### While printing")
        st.markdown(
            "- Load the barrel with **no air gaps**; pre-extrude before every print.\n"
            "- Big flat bases dry unevenly → mist lightly or use the staggered base.\n"
            "- Steep overhangs/folds sag — **stiffer clay** + **Fold softening** help.")
        st.markdown(f"**Placeholder** — clay mixing & wedging video · [RandoTechNerd]({LINK_RTN})")

    with tab_video:
        st.markdown("#### Video tour & how-tos")
        st.markdown(
            "- **Software tour** — _placeholder, coming soon_\n"
            "- **First print in 4 clicks** — _placeholder_\n"
            "- **EASY-mode setup (part list + assembly, clear-shot)** — _placeholder_\n"
            "- **Clay mixing & the slap test** — _placeholder_\n"
            "- **Cura / Orca setup for extra models** — _placeholder_")
        st.link_button("RandoTechNerd on YouTube", LINK_RTN,
                       icon=":material/smart_display:", use_container_width=True)

    with tab_plus:
        st.markdown("#### Support ClayShaper")
        st.markdown(
            ":material/check_circle: **Most of ClayShaper is free, and will stay "
            "free.** Design, slice, validate, upload your own STLs, print — the "
            "browser app is free for everyone, forever.")
        st.markdown(
            "#### :material/local_cafe: Early-adopter lifetime license — $99")
        st.markdown(
            "**[Buy me a coffee for $99]({url})** and I'll get you set up with the "
            "**local desktop version** — runs offline, on your own machine, yours "
            "to keep.".format(url=LINK_PLUS))
        st.link_button("Become an early adopter — $99 lifetime", LINK_PLUS,
                       icon=":material/local_cafe:", use_container_width=True)
        st.caption(
            f":material/mail: **Leave your email in the Buy Me a Coffee message/"
            f"note at checkout** and I'll send your local download. (Or email "
            f"**{CONTACT_EMAIL}** with your receipt.)")
        st.caption(
            "**Lifetime means lifetime** — we will never disable working software. "
            "This early-adopter price won't last, so it's the best deal ClayShaper "
            "will ever offer.")
        st.markdown(
            "#### :material/volunteer_activism: Just want to support?")
        st.markdown(
            f"Any [donation on Buy Me a Coffee]({LINK_DONATE}) helps fund "
            "development. **Supporters get early access to new features as they "
            "land, scaled to your donation** — the more you chip in, the more you "
            "unlock first. Most features will end up free for everyone; supporting "
            "just gets you there sooner and keeps the project alive.")
        st.caption(
            "Plus would genuinely help but $99 is out of reach? Email "
            "**randotechnerd@gmail.com** — we'll work something out.")


@st.dialog("Add your printer — BETA")
def add_printer_dialog():
    st.caption(
        "**BETA** — this builds a working local profile with a *generic* clay "
        "start-up (cold extrusion on, no brand-specific codes) and Eazao-class "
        "speed limits. It lives in this browser session. **Watch your first "
        "print closely** and tell us how it goes!")
    c_b, c_m = st.columns(2)
    with c_b:
        brand_in = st.text_input("Brand *", placeholder="e.g. Cerambot, StoneFlower")
    with c_m:
        model_in = st.text_input("Model *", placeholder="e.g. Eco Pro")
    c1, c2, c3 = st.columns(3)
    with c1:
        bx = st.number_input("Bed width X (mm) *", 50.0, 2000.0, 200.0, 5.0)
    with c2:
        by = st.number_input("Bed depth Y (mm) *", 50.0, 2000.0, 200.0, 5.0)
    with c3:
        mz = st.number_input("Max height Z (mm) *", 50.0, 2000.0, 250.0, 5.0)
    noz_txt = st.text_input("Nozzle sizes (mm, comma-separated)", "1.5, 2.0, 3.0")
    notes = st.text_area(
        "Anything else about it",
        placeholder="Extruder type (auger / ram / air pressure), firmware, "
                    "cartridge size, delta or cartesian…")

    if st.button("Create printer", type="primary", use_container_width=True,
                 icon=":material/add:"):
        brand_v, model_v = brand_in.strip(), model_in.strip()
        if not brand_v or not model_v:
            st.error("Brand and model are required — they name the profile and "
                     "let us add official support for your machine.")
        else:
            try:
                nozzles = sorted({round(float(v), 2) for v in
                                  noz_txt.replace(";", ",").split(",") if v.strip()})
            except ValueError:
                nozzles = []
            if not nozzles:
                nozzles = [2.0, 3.0]
            name = f"{brand_v} {model_v}"
            # Generic start-up: the Eazao sequence minus its dual-motor mix codes.
            generic_start = "\n".join(
                ln for ln in EAZAO_START_GCODE.splitlines()
                if not ln.startswith(("M163", "M164")))
            st.session_state.custom_printers[name] = {
                "bed_x": float(bx), "bed_y": float(by), "max_z": float(mz),
                "center_x": float(bx) / 2, "center_y": float(by) / 2,
                "print_speed": 1500, "z_speed": 300,
                "min_print_speed": 600, "max_print_speed": 2400,
                "max_feedrate": 3600, "layer_range": (0.4, 1.0),
                "cartridge_ml": 500.0, "filament_dia": 1.75,
                "nozzles": nozzles,
                "start_gcode": generic_start, "end_gcode": EAZAO_END_GCODE,
                "custom": True, "notes": notes.strip(),
            }
            _save_custom_printers(st.session_state.custom_printers)
            st.session_state.printer_name = name
            st.session_state.new_printer_done = {
                "name": name,
                "subj": _q(f"ClayShaper printer request: {name}"),
                "body": _q(f"Brand: {brand_v}\nModel: {model_v}\n"
                           f"Bed: {bx:g} x {by:g} mm\nMax height: {mz:g} mm\n"
                           f"Nozzles: {', '.join(f'{n:g}' for n in nozzles)} mm\n"
                           f"Notes: {notes.strip() or '-'}\n"),
            }

    # Success block lives OUTSIDE the button branch (a button nested inside
    # another button's branch never fires in Streamlit).
    _done = st.session_state.get("new_printer_done")
    if _done:
        st.success(f"**{_done['name']}** created and selected — it's now in "
                   "your printer list (marked BETA) and saved on this device.")
        st.markdown(
            "**One more thing — help us support it officially:** "
            f"[email us these details](mailto:randotechnerd@gmail.com?"
            f"subject={_done['subj']}&body={_done['body']}) "
            "(pre-filled, one click) and we'll build a tuned, tested profile "
            "for your machine.")
        if st.button("Done — back to the app", use_container_width=True):
            st.session_state.pop("new_printer_done", None)
            st.rerun()


# Opened by the "+ Add your printer" item in the machine dropdown (?printer=__add__).
if st.session_state.pop("show_add_printer", False):
    add_printer_dialog()


def _rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

ACC_RGB = _rgb(ACC)

# Custom CSS — ClayShaper brand
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, label, div {{
        font-family: 'Poppins','Segoe UI',sans-serif;
    }}
    h1, h2, h3, h4 {{ font-family: 'Poppins','Segoe UI',sans-serif; color: {INKC}; }}

    /* Whole-app theme: background, sidebar and base text colors */
    .stApp {{ background: {APP_BG}; color: {INKC}; }}
    section[data-testid="stSidebar"] {{ background: {SIDE_BG}; }}
    /* Widen ONLY when expanded — forcing width unconditionally broke collapse
       (the panel couldn't shrink, leaving a half-open sliver + stray controls). */
    section[data-testid="stSidebar"][aria-expanded="true"] {{
        width: 400px !important; min-width: 400px !important;
    }}
    [data-testid="stCaptionContainer"], .stCaption {{ color: {MUTEDC}; }}

    /* Pull content up: Streamlit's default top padding wastes a lot of room */
    [data-testid="stMainBlockContainer"] {{ padding-top: 1.1rem; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 0.6rem; }}

    /* Hide the hover link-anchor icons Streamlit puts next to every heading */
    [data-testid="stHeaderActionElements"] {{ display: none !important; }}

    /* Hide the top decoration bar and deploy button — but NEVER the sidebar
       expand/collapse controls, or a collapsed sidebar can't be reopened. */
    header[data-testid="stHeader"] {{ background: transparent; }}
    .stDeployButton, div[data-testid="stDecoration"], div[data-testid="stToolbar"] {{
        visibility: hidden;
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] * {{
        visibility: visible !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{ opacity: 1 !important; }}
    /* Expand (») affordance: visible ONLY while the sidebar is collapsed —
       forcing it always-on left a stray chevron floating next to the header. */
    .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stExpandSidebarButton"],
    .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stExpandSidebarButton"] * {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stExpandSidebarButton"] {{
        background: {SIDE_BG}; border: 1px solid {LINEC}; border-radius: 8px;
    }}
    .stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stExpandSidebarButton"] {{
        display: none !important;
    }}

    /* Main work area fills the screen */
    [data-testid="stMainBlockContainer"] {{
        max-width: 100%;
        padding-left: 2.2rem; padding-right: 2.2rem;
    }}

    /* Brand block (top-left) — tight, so the work area starts high */
    .ez-brand {{ margin: 0; line-height: 1; }}
    .ez-brand a {{ text-decoration: none; }}
    .ez-brand .ez-tagline {{
        text-transform: uppercase; letter-spacing: 4.6px; font-size: 0.72rem;
        font-weight: 600; color: {ACC_DK}; margin: -9px 0 0 5px;
    }}

    /* Numbered workflow steps (Model -> Slice -> Preview & Export) */
    .ez-step {{
        display: flex; align-items: center; gap: 10px; font-weight: 700;
        font-size: 1.02rem; color: {INKC}; margin: 12px 0 4px 0;
    }}
    .ez-step .ez-stepnum {{
        display: inline-flex; width: 26px; height: 26px; border-radius: 50%;
        background: {ACC}; color: #fff; align-items: center; justify-content: center;
        font-size: 0.82rem; font-weight: 700; flex: 0 0 26px;
    }}
    .ez-step .ez-stepsub {{ font-weight: 400; font-size: 0.8rem; color: {MUTEDC}; }}

    /* Printer dropdown (top of the sidebar settings panel) */
    .ez-dd {{ position: relative; display: block; width: 100%; margin: 10px 0 6px 0; }}
    .ez-dd summary {{ width: 100%; box-sizing: border-box; }}
    .ez-dd .ez-dd-menu {{ width: 100%; box-sizing: border-box; min-width: 0; }}
    .ez-dd summary {{
        list-style: none; cursor: pointer; display: flex; align-items: center;
        gap: 12px; border: 1.5px solid {LINEC}; border-radius: 12px;
        padding: 8px 16px; background: #fff; user-select: none;
    }}
    .ez-dd summary::-webkit-details-marker {{ display: none; }}
    .ez-dd summary:hover, .ez-dd[open] summary {{ border-color: {ACC}; }}
    .ez-dd .ez-ic {{ width: 44px; height: 40px; }}
    .ez-dd .ez-dd-name {{ font-weight: 600; font-size: 1.0rem; color: {INKC}; }}
    .ez-dd .ez-dd-sub {{ font-size: 0.72rem; color: {MUTEDC}; }}
    .ez-dd .ez-caret {{ margin-left: 6px; color: {ACC_DK}; font-size: 0.8rem; }}
    .ez-dd-menu {{
        position: absolute; top: calc(100% + 8px); left: 0; z-index: 1000;
        min-width: 300px; background: #fff; border: 1px solid {LINEC};
        border-radius: 14px; box-shadow: 0 10px 30px rgba(26,26,26,0.14); padding: 8px;
    }}
    .ez-dd-item, .ez-dd-item:hover, .ez-dd-item * {{
        text-decoration: none !important;
    }}
    .ez-dd-item {{
        display: flex; align-items: center; gap: 12px; padding: 9px 10px;
        border-radius: 10px;
    }}
    .ez-dd-item:hover {{ background: {SIDE_BG}; }}
    .ez-dd-item .ez-ic {{ width: 40px; height: 36px; }}
    .ez-dd-item.soon {{ opacity: 0.5; pointer-events: none; }}
    .ez-dd-item .ez-chip {{
        margin-left: auto; font-size: 0.6rem; font-weight: 700; letter-spacing: 1px;
        padding: 3px 8px; border-radius: 20px;
    }}
    .ez-dd-item .ez-chip.sel {{ background: {ACC}; color: #fff; }}
    .ez-dd-item .ez-chip.soon {{ border: 1px solid {LINEC}; color: {MUTEDC}; }}

    /* Section header labels in the sidebar */
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        text-transform: uppercase; letter-spacing: 1.6px; font-size: 0.9rem;
        color: {ACC_DK};
    }}

    /* Primary buttons -> accent, squared like their store */
    .stButton > button[kind="primary"], .stDownloadButton > button {{
        background: {ACC}; border: none; border-radius: 4px;
        color: #fff; font-weight: 600; letter-spacing: 0.5px;
    }}
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {{
        background: {ACC_DK}; color: #fff;
    }}

    /* Mode strip: boxed segmented tiles — connected squares in one bordered
       tray, selected tile tinted + outlined (ceramic-tile look). */
    [data-testid="stButtonGroup"] {{
        display: inline-flex; gap: 3px; padding: 4px;
        border: 1.5px solid {LINEC}; border-radius: 8px; background: #FFFFFF;
    }}
    [data-testid="stButtonGroup"] button {{
        border: 1.5px solid transparent !important; background: transparent !important;
        border-radius: 5px !important; padding: 9px 20px !important;
        font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;
        font-size: 0.82rem; color: {MUTEDC}; margin: 0 !important;
    }}
    [data-testid="stButtonGroup"] button:hover {{
        background: rgba({ACC_RGB}, 0.07) !important;
    }}
    [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {{
        color: {ACC_DK}; background: rgba({ACC_RGB}, 0.13) !important;
        border-color: {ACC} !important;
    }}

    /* Ceramic-tile pass: squared corners + crisp thin borders everywhere */
    .stButton button, .stDownloadButton button, .stLinkButton a,
    [data-testid="stFileUploaderDropzone"],
    [data-baseweb="select"] > div, .stTextInput input,
    [data-testid="stNumberInputContainer"],
    [data-testid="stAlert"], .stTextArea textarea {{
        border-radius: 6px !important;
    }}
    .stButton button, .stDownloadButton button, .stLinkButton a {{
        border-width: 1.5px !important; font-weight: 600 !important;
    }}
    button[data-testid="stBaseButton-primary"] {{
        border-radius: 6px !important; font-weight: 700 !important;
        letter-spacing: 0.4px;
    }}
    [data-testid="stExpander"] details {{ border-radius: 8px !important; }}
    .ez-dd summary, .ez-dd .ez-dd-menu {{ border-radius: 8px !important; }}

    /* Clay-brown scrollbars — but ONLY where content actually overflows.
       Several BaseWeb widget internals set overflow:scroll with nothing to
       scroll, which reserves an empty grey channel. So: zero-width bars
       everywhere by default, re-enabled just for the sidebar and the work
       area, where overflow:auto means they appear only when needed. */
    * {{ scrollbar-width: none; }}
    ::-webkit-scrollbar {{ width: 0; height: 0; }}

    [data-testid="stSidebarContent"], section[data-testid="stMain"] {{
        scrollbar-width: thin; scrollbar-color: {ACC} {LINEC};
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar,
    section[data-testid="stMain"]::-webkit-scrollbar {{ width: 11px; height: 11px; }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-track,
    section[data-testid="stMain"]::-webkit-scrollbar-track {{ background: transparent; }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb,
    section[data-testid="stMain"]::-webkit-scrollbar-thumb {{
        background: {ACC}; border-radius: 7px;
        border: 3px solid transparent; background-clip: content-box;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover,
    section[data-testid="stMain"]::-webkit-scrollbar-thumb:hover {{
        background: {ACC_DK}; background-clip: content-box;
    }}

    /* Slider tick labels only. Widget fills (slider track/thumb, checkmarks,
       toggles) come from the theme primaryColor set at mount time — do NOT
       hand-paint them here: an [aria-checked] rule matched the toggle's whole
       label wrapper and highlighted the text block behind it. */
    [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {{ color: {MUTEDC}; }}

    /* Slightly smaller default zoom so the workspace fits without scrolling. */
    html {{ zoom: 0.9; }}

    /* Compact slice-settings card: tighter gaps so checkboxes + sliders align */
    .st-key-slice_settings {{ padding: 6px 14px 10px 14px; }}
    .st-key-slice_settings [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
    .st-key-slice_settings [data-testid="stMarkdownContainer"] p {{
        margin-bottom: 9px; font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 1px; color: {MUTEDC};
    }}
    .st-key-slice_settings [data-testid="stCheckbox"] p,
    .st-key-slice_settings [data-testid="stWidgetLabel"] p {{
        text-transform: none !important; letter-spacing: 0 !important;
        font-size: 0.92rem; color: {INKC};
    }}
    .st-key-slice_settings [data-testid="stSlider"] {{ padding-top: 2px; }}
    /* Number input: keep container, field and stepper buttons the same height
       so the -/+ controls sit flush inside the box. */
    .st-key-slice_settings [data-testid="stNumberInputContainer"],
    .st-key-slice_settings [data-testid="stNumberInputContainer"] > div,
    .st-key-slice_settings [data-testid="stNumberInputContainer"] button,
    .st-key-slice_settings [data-testid="stNumberInputField"] {{
        height: 36px; min-height: 36px;
    }}
    .st-key-slice_settings [data-testid="stNumberInputContainer"] {{
        align-items: stretch; overflow: hidden; border-radius: 8px;
    }}

    /* Compact "+" replace-model button (shown once a model is loaded) */
    .st-key-replace_model {{ margin-left: 26px; margin-top: 8px; }}
    .st-key-replace_model button {{
        width: 54px; height: 54px; border: 2px dashed {ACC}; border-radius: 12px;
        background: rgba({ACC_RGB}, 0.06); color: {ACC}; font-size: 1.6rem;
        font-weight: 300; line-height: 1;
    }}
    .st-key-replace_model button:hover {{
        background: rgba({ACC_RGB}, 0.14); border-color: {ACC_DK}; color: {ACC_DK};
    }}
    .ez-model-chip {{
        display: inline-flex; align-items: center; gap: 12px; margin-top: 2px;
        font-weight: 600; color: {INKC};
    }}
    /* Little "window" showing what's loaded (sliced thumbnail of the model) */
    .ez-chip-thumb {{
        width: 56px; height: 56px; border-radius: 10px; object-fit: cover;
        border: 1px solid {LINEC}; background: #fff; flex: 0 0 56px;
    }}
    .ez-chip-noimg {{ display: inline-flex; align-items: center; justify-content: center; }}
    .ez-model-chip .ez-chip-txt {{ display: flex; flex-direction: column; line-height: 1.25; }}
    .ez-model-chip .ez-sub {{ font-size: 0.75rem; color: {MUTEDC}; font-weight: 400; }}


    /* Big "+" drop-target for the STL uploader (scoped via container key).
       The Browse button is stretched invisibly over the whole zone, so a click
       anywhere opens the file picker; drag-and-drop hits the same zone. */
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzone"] {{
        position: relative;
        min-height: 230px;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        gap: 0;
        border: 2px dashed {ACC};
        border-radius: 14px;
        background: rgba({ACC_RGB}, 0.06);
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
    }}
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzone"]:hover {{
        background: rgba({ACC_RGB}, 0.14);
        border-color: {ACC_DK};
    }}
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzone"]::before {{
        content: "+";
        font-size: 84px;
        font-weight: 300;
        line-height: 1;
        color: {ACC};
    }}
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzone"]::after {{
        content: "Drop your STL here — or click anywhere to browse";
        font-size: 1.05rem;
        color: {MUTEDC};
        margin-top: 10px;
    }}
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none;
    }}
    .st-key-stl_dropzone [data-testid="stFileUploaderDropzone"] button {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
    }}

    /* Site footer (contact + links), shown at the bottom of every mode */
    .cs-footer {{
        margin: 3.5rem 0 0.5rem;
        padding-top: 1.4rem;
        border-top: 1px solid {LINEC};
        text-align: center;
    }}
    .cs-foot-brand {{ opacity: 0.85; margin-bottom: 0.5rem; }}
    .cs-foot-brand svg {{ height: 20px; }}
    .cs-foot-links {{ font-size: 0.9rem; margin-bottom: 0.35rem; }}
    .cs-foot-links a {{
        color: {ACC};
        text-decoration: none;
        font-weight: 500;
    }}
    .cs-foot-links a:hover {{ text-decoration: underline; }}
    .cs-foot-dot {{ color: {MUTEDC}; margin: 0 0.55rem; }}
    .cs-foot-fine {{ font-size: 0.76rem; color: {MUTEDC}; }}
</style>
""", unsafe_allow_html=True)

# --- Brand block (top-left): big clickable wordmark + tagline. The printer
# --- dropdown lives in the sidebar settings panel (Orca-style).
st.markdown(f"""
<div class="ez-brand">
  <a href="{LINK_HOME}" target="_blank" title="clayshaper.com">{wordmark_svg(64)}</a>
  <div class="ez-tagline">Ceramic Slicer</div>
</div>
""", unsafe_allow_html=True)


_PLUS_ICON = (
    '<svg viewBox="0 0 90 80" xmlns="http://www.w3.org/2000/svg" fill="none" '
    'stroke="#B0603C" stroke-width="4" stroke-linecap="round">'
    '<path d="M45 22 V58 M27 40 H63"/></svg>')


def printer_dropdown_html():
    """The machine picker: a box with icon + name + caret that drops open."""
    items = []
    for name, _svg_fn, tagline, active in PRINTER_CATALOG:
        sel = (name == selected_printer)
        if active:
            # Web build: no ?printer= navigation (it would remount Pyodide);
            # selection happens via the native picker below the dropdown.
            items.append(
                f'<div class="ez-dd-item">'
                f'<span class="ez-ic">{printer_icon(name, active=True)}</span>'
                f'<span><div class="ez-dd-name">{name}</div>'
                f'<div class="ez-dd-sub">{tagline}</div></span>'
                + ('<span class="ez-chip sel">SELECTED</span>' if sel else '')
                + '</div>'
            )
        else:
            items.append(
                f'<div class="ez-dd-item soon">'
                f'<span class="ez-ic">{printer_icon(name, active=False)}</span>'
                f'<span><div class="ez-dd-name">{name}</div>'
                f'<div class="ez-dd-sub">{tagline}</div></span>'
                f'<span class="ez-chip soon">SOON</span></div>'
            )
    # User-added printers (BETA) live under the catalog machines.
    for name, prof in st.session_state.custom_printers.items():
        sel = (name == selected_printer)
        items.append(
            f'<div class="ez-dd-item">'
            f'<span class="ez-ic">{printer_icon(name, active=True)}</span>'
            f'<span><div class="ez-dd-name">{name}</div>'
            f'<div class="ez-dd-sub">{prof["bed_x"]:g}×{prof["bed_y"]:g}×{prof["max_z"]:g} · your printer</div></span>'
            + ('<span class="ez-chip sel">SELECTED</span>' if sel else '<span class="ez-chip soon">BETA</span>')
            + '</div>'
        )
    # (Web build: "+ Add your printer" is a native button under the dropdown.)
    cfg = ALL_PROFILES[selected_printer]
    return f"""
<details class="ez-dd">
  <summary>
    <span class="ez-ic">{printer_icon(selected_printer, active=True)}</span>
    <span>
      <div class="ez-dd-name">{selected_printer}</div>
      <div class="ez-dd-sub">Bed {cfg['bed_x']:g}×{cfg['bed_y']:g} · max Z {cfg['max_z']:g} mm</div>
    </span>
    <span class="ez-caret">▼</span>
  </summary>
  <div class="ez-dd-menu">{''.join(items)}</div>
</details>"""


# --- Settings header: Orca-style tab strip on top of the work area ---
# Slice STL first: a new user can drop a model (or grab a sample) and print.
st.session_state.setdefault("mode_tabs", "Slice STL")
mode = st.segmented_control(
    "Mode", ["Slice STL", "Design", "Validate G-code"],
    key="mode_tabs", label_visibility="collapsed",
) or "Slice STL"

# --- SIDEBAR (settings panel, Orca-style) ---
with st.sidebar:
    # (Sidebar wordmark removed — the big ClayShaper brand up top covers it;
    # this gives the settings panel more vertical room.)

    # Machine dropdown (icon + name + caret) — the single place the printer lives.
    printer_name = selected_printer
    profile = dict(ALL_PROFILES[printer_name])
    st.markdown(printer_dropdown_html(), unsafe_allow_html=True)

    # Web build: native add/select (no URL navigation under stlite).
    if st.button("Add your printer — BETA", icon=":material/add:",
                 use_container_width=True, key="add_printer_btn"):
        st.session_state.show_add_printer = True
        st.rerun()
    if st.session_state.custom_printers:
        _opts = list(ALL_PROFILES.keys())
        _pick = st.selectbox("Active printer", _opts,
                             index=_opts.index(printer_name) if printer_name in _opts else 0,
                             key="printer_pick")
        if _pick != printer_name:
            st.session_state.printer_name = _pick
            st.rerun()

    st.divider()
    st.header("Process")

    # Nozzle Selection with Visual Color Swatch.
    # Eazao's shipped profiles: 3mm line / 1.0mm layer (vase) and 2mm / 0.6mm
    # (infill) — so 3.0 is the factory default.
    c_noz_sel, c_noz_col = st.columns([3, 1])

    with c_noz_sel:
        nozzle_options = profile["nozzles"]
        _default_noz = nozzle_options.index(3.0) if 3.0 in nozzle_options else 0
        nozzle = st.selectbox(
            "Nozzle / line width (mm)", nozzle_options,
            index=_default_noz, key="s_noz",
            help="Eazao runs line width equal to the nozzle bore. Drives extrusion volume.",
        )

    # Keep typed/stored slider values inside their (nozzle-dependent) ranges.
    _lh_max = round(nozzle * 0.9, 2)
    if "s_lh" in st.session_state:
        st.session_state.s_lh = min(max(st.session_state.s_lh, 0.2), _lh_max)
    _flh_max = round(max(st.session_state.get("s_lh", min(1.0, round(nozzle * 0.5, 2))) * 1.5, 0.6), 2)
    if "s_flh" in st.session_state:
        st.session_state.s_flh = min(max(st.session_state.s_flh, 0.3), _flh_max)
    if "s_flf" in st.session_state:
        st.session_state.s_flf = int(min(max(st.session_state.s_flf, 80), 200) // 5 * 5)

    # Color-code the nozzle by relative size (small = pink … large = blue)
    nmin, nmax = min(nozzle_options), max(nozzle_options)
    frac = 0.0 if nmax == nmin else (nozzle - nmin) / (nmax - nmin)
    palette = ["#FF4081", "#9E9E9E", "#4CAF50", "#2196F3"]
    nozzle_color_var = palette[min(int(frac * (len(palette) - 1) + 0.5), len(palette) - 1)]

    with c_noz_col:
        st.write("")  # Spacing
        st.write("")
        dot_size = int(16 + frac * 24)  # 16px (smallest) … 40px (largest)
        st.markdown(f"""
        <div style="
            background-color: {nozzle_color_var};
            width: {dot_size}px;
            height: {dot_size}px;
            border-radius: 50%;
            margin-top: {20 - dot_size/2}px;
            border: 2px solid #333;
        " title="Nozzle {nozzle}mm"></div>
        """, unsafe_allow_html=True)

    # Factory default layer height: 1.0mm at 3mm nozzle (Eazao vase profile),
    # capped by the machine's spec range for smaller nozzles.
    # Keyed sliders are seeded ONCE via session state and take no `value`
    # param — passing both makes Streamlit warn ("created with a default value
    # but also had its value set via the Session State API") because the typed-
    # entry Apply button also writes these keys.
    st.session_state.setdefault("s_lh", min(1.0, round(nozzle * 0.5, 2)))
    layer_h = st.slider(
        "Layer Height (mm)", 0.2, _lh_max, key="s_lh",
        help="Eazao spec: 0.4–1.0 mm. Factory vase profile uses 1.0 mm.",
    )

    st.session_state.setdefault("s_flh", layer_h)
    first_layer_h = st.slider(
        "First layer height (mm)", 0.3, round(max(layer_h * 1.5, 0.6), 2), step=0.05,
        key="s_flh",
        help="Height of layer 1 only. Lower squishes the first coil into the bed "
             "for grip; the extrusion volume adjusts automatically.",
    )

    st.session_state.setdefault("s_flf", 100)
    flf_pct = st.slider(
        "First layer flow (%)", 80, 200, step=5, key="s_flf",
        help="Extra clay on layer 1 for bed adhesion. 100% = same as the rest; "
             "try 110–130% if the first coil doesn't stick.",
    )
    first_layer_flow = flf_pct / 100.0

    if mode == "Design":
        st.divider()
        st.header("Shape Definition")
        shape_type = st.selectbox("Profile", ["Cylinder", "Bowl", "Vase", "Virtual Wheel", "Handle"])

        # Virtual Wheel variables (initialized here, processed in main column)
        custom_profile_data = None

        if shape_type == "Handle":
            # Handles print FLAT on the bed: the curve lies in the bed plane
            # and the strap width becomes the print height. Attach when
            # leather-hard (score + slip).
            st.subheader("Handle")
            handle_style = st.selectbox("Style", HANDLE_STYLES,
                help="All shapes print as one continuous bead. Kuksa = a "
                     "double-lobed grip; Hook (7) = classic open-bottom handle.")
            handle_w = st.slider("Width — between the attachment ends (mm)",
                                 20.0, 120.0, 60.0, 2.0)
            if handle_style == "Half Circle":
                handle_h = handle_w / 2.0
                st.caption(f"Reach: {handle_h:.0f} mm (half the width — it's a circle)")
            elif handle_style == "Half Square":
                handle_h = handle_w
                st.caption(f"Reach: {handle_h:.0f} mm (same as width — it's a square)")
            else:
                handle_h = st.slider("Reach — how far it sticks out (mm)",
                                     10.0, 90.0, 35.0, 2.0)
            kuksa_dip = 100
            if handle_style == "Kuksa (Double)":
                kuksa_dip = st.slider(
                    "Center point — dip toward the pot (%)", 0, 100, 100, 5,
                    help="100% = the middle of the two loops touches the pot "
                         "(classic kuksa, like two circles side by side). "
                         "Lower pulls the center point outward for a "
                         "shallower valley.")
            strap_w = st.slider("Strap width (mm)", 6.0, 40.0, 16.0, 1.0,
                help="The width of the strap you'll grip — printed as height "
                     "on the bed.")
            hd_beads = st.selectbox("Strap thickness (beads)", [1, 2, 3, 4, 5, 6],
                index=0,
                help="Parallel passes side by side — each bead ≈ your nozzle "
                     "width. Traced as one continuous serpentine, so any "
                     "thickness still prints without a single stop.")
            handle_copies = st.number_input("Copies", 1, 4, 1,
                help="Print several at once — stack two on a mug for a "
                     "multi-finger hold, or batch for a set of cups.")
            # Defaults for downstream code paths shared with vessels.
            roundness = 90
            body_base_radius = handle_w
            height = strap_w
            base_h = 0.0
            base_bottom_radius = None
            tex_type = "None"
            tex_amp, tex_freq, twist_factor = 0.0, 10.0, 0.0
            img_array = None
        else:
            roundness = st.slider(
                "Roundness (segments / layer)", 24, 180, 90, step=6,
                help="Points per revolution. Higher = smoother walls (less faceting on wide pots), but more G-code.",
            )

            if shape_type == "Virtual Wheel":
                st.subheader("Dimensions")
                body_base_radius = st.slider("Max Radius (Width) (mm)", 20.0, 150.0, 60.0)
                height = st.slider("Total Height (mm)", 20.0, 400.0, 120.0)

            else:
                st.subheader("Main Body")
                body_base_radius = st.slider("Body Base Radius (mm)", 20.0, 150.0, 60.0)
                height = st.slider("Total Height (mm)", 20.0, 400.0, 120.0)

            st.subheader("Solid Base")
            base_h = st.slider("Base Height (mm)", 0.0, 20.0, 4.0, step=layer_h)

        # By default the base footprint auto-matches the vessel's real bottom
            # radius (crucial for Virtual Wheel shapes with narrow feet).
            custom_base_r = st.checkbox(
                "Custom base radius", value=False,
                help="Off = the solid base exactly matches the vessel's bottom. "
                     "On = set it yourself (e.g. a wider foot for stability).")
            base_bottom_radius = None
            if custom_base_r:
                base_bottom_radius = st.slider("Base Bottom Radius (mm)", 0.0, body_base_radius + 50, body_base_radius)

            st.divider()
            st.header("Texture (Imprint)")
            tex_type = st.selectbox("Pattern", ["None", "Sine Waves", "Twist", "Image Map"])

            tex_amp = 0.0
            tex_freq = 10.0

            # Image Map specific variables
            img_array = None

            if tex_type == "Image Map":
                st.info("Upload an image to wrap around the vessel. Darker pixels = inward, Lighter = outward.")
                uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
                tex_amp = st.slider("Max Amplitude (mm)", 0.0, 30.0, 5.0)

                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('L')
                    # Resize for performance - we don't need massive resolution for clay
                    image.thumbnail((500, 500))
                    st.image(image, caption="Texture Map", width=200)

                    # Convert to numpy and normalize
                    img_array = np.array(image).astype(float) / 255.0

            elif tex_type != "None":
                tex_amp = st.slider("Amplitude (mm)", 0.0, 30.0, 3.0)
                tex_freq = st.slider("Frequency/Ridges", 1, 60, 12)

            twist_factor = 0.0
            if tex_type == "Twist":
                twist_factor = st.slider("Twist Rotations", 0.0, 5.0, 1.0)

    # Bed & placement — defaults come from the printer profile.
    st.divider()
    with st.expander("Advanced", icon=":material/tune:"):
        st.markdown("**Fine-tune values**")
        st.caption("Type exact heights and flow instead of using the sliders.")
        t_lh = st.number_input("Layer height (mm)", 0.2, _lh_max,
                               st.session_state.get("s_lh", min(1.0, round(nozzle * 0.5, 2))),
                               0.01, key="t_lh")
        t_flh = st.number_input("First layer height (mm)", 0.3, 3.0,
                                st.session_state.get("s_flh", st.session_state.get("s_lh", 1.0)),
                                0.01, key="t_flh")
        t_flf = st.number_input("First layer flow (%)", 80, 200,
                                st.session_state.get("s_flf", 100), 5, key="t_flf")

        def _apply_typed():
            noz = st.session_state.get("s_noz", 3.0)
            lh_max = round(noz * 0.9, 2)
            lh = min(max(st.session_state.t_lh, 0.2), lh_max)
            st.session_state.s_lh = round(lh, 2)
            flh_max = round(max(lh * 1.5, 0.6), 2)
            st.session_state.s_flh = round(min(max(st.session_state.t_flh, 0.3), flh_max), 2)
            st.session_state.s_flf = int(min(max(st.session_state.t_flf, 80), 200) // 5 * 5)

        st.button("Apply to sliders", type="primary", use_container_width=True,
                  on_click=_apply_typed)
        st.divider()

        st.markdown("**Bed & placement**")
        c_bed1, c_bed2 = st.columns(2)
        with c_bed1:
            profile["bed_x"] = st.number_input("Bed Width (X)", value=float(profile["bed_x"]))
            profile["center_x"] = st.number_input("Center X", value=float(profile["center_x"]))
        with c_bed2:
            profile["bed_y"] = st.number_input("Bed Depth (Y)", value=float(profile["bed_y"]))
            profile["center_y"] = st.number_input("Center Y", value=float(profile["center_y"]))
        st.caption("Toolpath is centered on (Center X, Center Y). Eazao machines are cartesian — leave at bed center.")

        st.markdown("**Appearance**")
        st.selectbox(
            "Theme", list(THEMES.keys()), key="theme_accent",
            help="Restyles the whole studio: accent, sidebar, background and text colors.",
        )
        c_clay, c_base = st.columns(2)
        with c_clay:
            st.color_picker("Clay color", value=CLAY, key="clay_color",
                            help="Color of the walls in the 3D previews.")
        with c_base:
            st.color_picker("Base color", value=BASE_CLR, key="base_color",
                            help="Color of the base layers in the previews. Set it "
                                 "to your clay color to make the whole print one color.")
        st.selectbox("Material units", ["grams", "kilograms", "pounds", "ml"],
                     key="mat_unit",
                     help="Units for the “Clay needed” estimate. It's an estimate "
                          "only — real usage varies with clay body and water content.")

        st.divider()
        st.markdown("**Help**")
        if st.button("Setup, Tips & Tricks", icon=":material/menu_book:",
                     use_container_width=True):
            setup_tips_dialog()


def render_footer():
    """Site footer — contact + links. Rendered at the bottom of every mode
    (each mode st.stop()s, so it's called before each stop, not once globally)."""
    yr = datetime.now().year
    st.markdown(f"""
<div class="cs-footer">
  <div class="cs-foot-brand">{wordmark_svg(20)}</div>
  <div class="cs-foot-links">
    <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
    <span class="cs-foot-dot">·</span>
    <a href="{LINK_RTN}" target="_blank" rel="noopener">YouTube · @RandoTechNerd</a>
    <span class="cs-foot-dot">·</span>
    <a href="{LINK_RTN_SITE}" target="_blank" rel="noopener">RandoTechNerd.com</a>
  </div>
  <div class="cs-foot-fine">A RandoTechNerd build · built for clay printers ·
     © {yr} RandoTechNerd</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MODE: Slice STL
# =============================================================================
if mode == "Slice STL":
    # (No mode heading — the tab strip above already names the mode. The
    # numbered steps below carry the flow, and this reclaims vertical room.)
    _samples = list_sample_stls()

    # --- Sample model picker: an image menu — the pictures ARE the buttons ---
    @st.dialog("Included sample models", width="large")
    def sample_picker():
        st.caption("Each model is shown exactly as it comes out of the slicer — "
                   "blue coils are the solid base, terracotta the walls. "
                   "**Click a picture to load it.**")
        if not _samples:
            st.warning("Sample folder not found (Dev/Eazao Potter/STL model).")
            return
        st.markdown("""
        <style>
            .cs-gallery { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
            .cs-card {
                display: block; text-decoration: none; border: 1px solid #E4D5C7;
                border-radius: 12px; overflow: hidden; background: #FAF8F5;
                transition: transform 0.12s, box-shadow 0.12s, border-color 0.12s;
            }
            .cs-card:hover {
                transform: translateY(-3px); border-color: #C17C53;
                box-shadow: 0 6px 18px rgba(160, 90, 53, 0.25);
            }
            .cs-card img { width: 100%; display: block; }
            .cs-card .cs-name {
                text-align: center; padding: 7px 4px; font-weight: 600;
                color: #6B4F3A; font-size: 0.95rem;
            }
            .cs-cat { margin: 14px 0 6px 0; font-weight: 700; color: #4a4a4a; }
        </style>
        """, unsafe_allow_html=True)

        # Web build: native buttons instead of ?sample= links — a URL
        # navigation would remount the whole Python runtime under stlite.
        by_cat = {}
        for name, cat, path in _samples:
            by_cat.setdefault(cat, []).append((name, path))
        for cat in ["Classics", "Vase", "Cartoon"]:
            if cat not in by_cat:
                continue
            st.markdown(f'<div class="cs-cat">{cat}</div>', unsafe_allow_html=True)
            items = by_cat[cat]
            for row in range(0, len(items), 3):
                cols = st.columns(3)
                for col, (name, path) in zip(cols, items[row:row + 3]):
                    with col:
                        b64 = get_thumbnail_b64(path)
                        if b64:
                            st.markdown(
                                f'<div class="cs-card"><img src="{b64}" alt="{name}"/></div>',
                                unsafe_allow_html=True)
                        if st.button(name, key=f"pick_{cat}_{name}",
                                     use_container_width=True):
                            st.session_state.current_model = ("sample", name, path)
                            st.session_state.mode_tabs = "Slice STL"
                            st.rerun()

    # --- STEP 1 · MODEL -------------------------------------------------------
    # While nothing is loaded, show the big drop-anywhere zone. Once a model is
    # in, collapse it to a small "+" button (click to replace).
    st.markdown('<div class="ez-step"><span class="ez-stepnum">1</span> Model</div>',
                unsafe_allow_html=True)
    current = st.session_state.get("current_model")   # (kind, name, payload)

    if current is None:
        with st.container(key="stl_dropzone"):
            stl_file = st.file_uploader("STL file", type=["stl"], key="stl_uploader",
                                        label_visibility="collapsed")
        if stl_file is not None:
            st.session_state.current_model = (
                "upload", stl_file.name.rsplit(".", 1)[0], stl_file.getvalue())
            st.rerun()
        if st.button("No STL? Browse the included sample models →",
                     type="tertiary", icon=":material/grid_view:"):
            sample_picker()
        model_name = None
    else:
        kind, model_name, payload = current
        c_plus, c_chip, c_scale, c_browse = st.columns([0.55, 2.4, 1.25, 1.5])
        with c_plus:
            if st.button(":material/add:", key="replace_model",
                         help="Load a different STL"):
                st.session_state.pop("current_model", None)
                st.session_state.pop("stl_uploader", None)
                st.rerun()
        with c_chip:
            src = "sample model" if kind == "sample" else "uploaded STL"
            thumb = model_thumb_b64(kind, model_name, payload)
            window = (f'<img class="ez-chip-thumb" src="{thumb}"/>' if thumb
                      else f'<span class="ez-chip-thumb ez-chip-noimg">{ui_icon("slice", 18, ACC)}</span>')
            st.markdown(
                f'<div class="ez-model-chip">{window}'
                f'<span class="ez-chip-txt">{model_name}'
                f'<span class="ez-sub">{src}</span></span></div>',
                unsafe_allow_html=True)
        with c_scale:
            scale_pct = st.number_input("Scale (%)", 10, 400, 100, 5, key="scale_pct",
                                        help="Resize the model before slicing. 100% = as modeled.")
        with c_browse:
            if st.button("Browse samples", type="tertiary",
                         icon=":material/grid_view:", key="browse_when_loaded"):
                sample_picker()

    # --- STEP 2 · SLICE SETTINGS ---------------------------------------------
    st.markdown('<div class="ez-step"><span class="ez-stepnum">2</span> Slice '
                '<span class="ez-stepsub">walls · base · quality</span></div>',
                unsafe_allow_html=True)
    with st.container(border=True, key="slice_settings"):
        sc1, sc2, sc3, sc4 = st.columns([1.2, 1.2, 1.3, 1.3], gap="medium")
        with sc1:
            st.markdown("**Walls**")
            vase_mode = st.checkbox("Vase mode (spiral)", value=True,
                                    help="Best for clay: one continuous bead per layer, no seams.")
            continuous = st.checkbox("Continuous bead", value=True, disabled=not vase_mode,
                                     help="Joins every layer to the next with extrusion instead of a "
                                          "travel — the whole vessel is ONE unbroken bead, so the "
                                          "nozzle can never stop or loop at the seam.")
            soften_choice = st.selectbox(
                "Fold softening", ["Off", "Gentle", "Medium", "Strong"],
                disabled=not vase_mode,
                help="Tames deep folds and steep sideways overhangs (like Orca/Cura's "
                     "'make overhang printable'): each layer is kept within a max "
                     "sideways step of the coil below, so material lands closer to "
                     "the previous layer. Softens sharp creases into printable ones.")
            _soften_map = {"Off": None, "Gentle": 1.5, "Medium": 1.0, "Strong": 0.6}
            fold_soften = _soften_map[soften_choice]
        with sc2:
            st.markdown("**Base**")
            staggered = st.checkbox("Staggered base", value=True,
                                    help="Offsets base rings on alternating layers so seams don't stack. "
                                         "In the preview, base layers alternate light/dark so you can see it.")
            bottom_layers = st.number_input("Base layers", 1, 20, 3)
        with sc3:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            stagger_off = st.slider("Stagger offset", 0.0, 1.0, 0.5, 0.05, disabled=not staggered,
                                    help="Fraction of a line width each alternating base layer shifts inward.")
            stagger_fill = st.slider("Inset fill", 1.0, 1.3, 1.1, 0.05, disabled=not staggered,
                                     help="Extra clay on the inset base layers so they bond to their "
                                          "neighbors — fills the gaps that make the base look like an "
                                          "under-filled Oreo. 1.1–1.15 is usually plenty.")
        with sc4:
            st.markdown("**Quality**")
            path_res = st.slider("Path resolution (mm)", 0.3, 5.0, 1.5, 0.1,
                                 help="Max segment length. Lower = smoother walls, larger files.")

    model_scale = st.session_state.get("scale_pct", 100) / 100.0
    slice_key = (model_name, nozzle, layer_h, int(bottom_layers) if model_name else 0,
                 staggered, stagger_off, vase_mode, path_res, printer_name,
                 first_layer_flow, first_layer_h, continuous, fold_soften, model_scale,
                 stagger_fill)

    do_slice = st.button("Slice", type="primary", icon=":material/play_arrow:",
                         use_container_width=True, disabled=(model_name is None))

    if do_slice and model_name is not None:
        kind, _, payload = st.session_state.current_model
        if kind == "upload":
            # In-memory upload: stage the stored bytes to a temp file for trimesh.
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp.write(payload)
                slice_path = tmp.name
        else:
            slice_path = payload  # sample: payload is the file path
        try:
            _diag = {}
            with st.spinner(f"Slicing {model_name}…"):
                gcode_str, layers = slice_stl(
                    slice_path, profile, nozzle=nozzle, layer_height=layer_h,
                    bottom_layers=int(bottom_layers), staggered=staggered,
                    staggered_offset_factor=stagger_off, vase_mode=vase_mode,
                    path_resolution=path_res, first_layer_flow=first_layer_flow,
                    first_layer_height=first_layer_h, continuous=continuous,
                    fold_softening=fold_soften, scale=model_scale, stagger_fill=stagger_fill,
                    diagnostics=_diag,
                    source=f'Sliced STL "{model_name}" ({kind}, {model_scale*100:.0f}%) on {printer_name}')
            st.session_state.slice_diag = _diag
            # Remember the settings THIS slice was made with: the preview must
            # render with these, not the live sliders (otherwise nudging Layer
            # Height after slicing draws thin slatted coils with fake gaps).
            st.session_state.slice_result = {"key": slice_key, "gcode": gcode_str,
                                             "layers": layers, "name": model_name,
                                             "layer_h": layer_h, "nozzle": nozzle}
        finally:
            if kind == "upload":
                os.unlink(slice_path)

    # --- STEP 3 · PREVIEW & EXPORT ---------------------------------------------
    result = st.session_state.get("slice_result")
    if model_name is None:
        st.info("Load a model above — upload an STL or pick an included sample — then press Slice.")
    elif result is None:
        st.info("Press **Slice** to generate the toolpath.")
    else:
        stale = result["key"] != slice_key
        st.markdown('<div class="ez-step"><span class="ez-stepnum">3</span> Preview & export '
                    '<span class="ez-stepsub">inspect the nozzle path, then download</span></div>',
                    unsafe_allow_html=True)
        if stale:
            st.warning("The model or settings changed since this slice — press **Slice** again to refresh.",
                       icon=":material/refresh:")

        # Partial slice? Say so loudly — a model that stops a third of the way
        # up otherwise looks like a deliberate (and valid) short print.
        _d = st.session_state.get("slice_diag") or {}
        if _d.get("failed_heights"):
            _short = _d["model_top_mm"] - _d["sliced_top_mm"] > max(2 * layer_h, 1.0)
            st.warning(
                f"**Only part of this model could be sliced.** "
                f"{_d['failed_heights']} height(s) couldn't be read from the mesh, "
                f"so the toolpath stops at {_d['sliced_top_mm']:.0f} mm of "
                f"{_d['model_top_mm']:.0f} mm."
                + ("  \nThis usually means the STL has holes, flipped normals or "
                   "self-intersections. Try **Vase mode**, or repair the mesh "
                   "(Blender / Meshmixer / netfabb) and re-upload."
                   if _short else "")
                + (f"  \n\n`{_d['last_error']}`" if _d.get("last_error") else ""),
                icon=":material/warning:")

        layers = result["layers"]
        gcode_str = result["gcode"]
        # Render with the settings the slice was MADE with, not the live sliders.
        res_lh = result.get("layer_h", layer_h)
        res_noz = result.get("nozzle", nozzle)
        n_layers = len(layers)

        lo, hi = st.slider(
            "Preview layers", 0, max(n_layers - 1, 1), (0, max(n_layers - 1, 1)),
            help="Narrow the range to inspect specific layers — e.g. just the base "
                 "to confirm the stagger.")

        # Shaded, Orca-style preview: each extrusion run becomes a lit ribbon
        # one layer tall, so the toolpath reads as a solid printed object.
        # Staggered base layers alternate two blues so the offset is visible.
        runs = {"base A": [], "base B": [], "walls": []}
        base_idx = 0
        for li, layer in enumerate(layers):
            is_base = (layer["type"] == "bottom")
            if is_base:
                key = "base A" if base_idx % 2 == 0 else "base B"
                base_idx += 1
            else:
                key = "walls"
            if not (lo <= li <= hi):
                continue
            for path in layer["paths"]:
                if path is None:
                    continue
                coords = np.asarray(path.coords)
                if len(coords) < 2:
                    continue
                if layer["type"] == "vase":
                    zline = np.linspace(layer["z"], layer["z"] + res_lh, len(coords))
                else:
                    zline = np.full(len(coords), layer["z"])
                runs[key].append((coords[:, 0], coords[:, 1], zline))

        # base A = chosen base color, base B = a lighter tint so the stagger
        # stays visible even when base color is set equal to the clay color.
        fig = toolpath_figure(
            [("base A", runs["base A"], BASE_CLR, True),
             ("base B", runs["base B"], _lighten(BASE_CLR), True),
             ("walls", runs["walls"], CLAY, False)],
            res_lh, profile["bed_x"], profile["bed_y"], height=560,
            bead_width=res_noz)
        # Stable key + uirevision keep the camera fixed when the layer range
        # slider changes — only the geometry updates, the zoom/angle stays put.
        st.plotly_chart(fig, use_container_width=True, key="slice_preview")

        with st.expander("Validation report", expanded=True, icon=":material/verified:"):
            render_validation(gcode_str, profile, nozzle=res_noz, layer_height=res_lh)

        st.download_button(
            "Download G-code", gcode_str,
            file_name=result["name"] + ".gcode", icon=":material/download:",
            mime="text/plain", use_container_width=True, type="primary",
            disabled=stale)
    render_footer()
    st.stop()


# =============================================================================
# MODE: Validate G-code
# =============================================================================
if mode == "Validate G-code":
    st.caption(f"Upload a .gcode file to check it against the {printer_name}: bounds, "
               "speeds, clay-safety start codes, and material usage.")
    gfile = st.file_uploader("G-code file", type=["gcode", "gco", "g", "nc", "txt"],
                             label_visibility="collapsed")

    # Or validate one of the factory files shipped with the printer.
    text = None
    source_name = None
    factory = list_sample_gcodes()
    if factory:
        names = ["— choose an included Eazao G-code —"] + [n for n, _ in factory]
        pick = st.selectbox("No file? Validate an included Eazao factory G-code:", names)
        if gfile is None and pick != names[0]:
            path = dict(factory)[pick]
            with open(path, errors="ignore") as f:
                text = f.read()
            source_name = pick

    if gfile is not None:
        text = gfile.getvalue().decode("utf-8", errors="ignore")
        source_name = gfile.name

    if text is not None:
        st.markdown(f"**Validating:** `{source_name}`")
        # Prefer the file's own declared layer height for rendering/validation
        # (the sidebar slider may be set for a different job).
        _m = re.search(r";Layer height:\s*([0-9.]+)", text)
        file_lh = float(_m.group(1)) if _m else layer_h
        vrep = render_validation(text, profile, nozzle=nozzle, layer_height=file_lh)

        # Save the validated file under a new name (e.g. "bowl_checked.gcode").
        default_stem = os.path.splitext(source_name)[0].strip() + "_validated"
        c_name, c_save = st.columns([3, 1.4])
        with c_name:
            save_stem = st.text_input("Save as", value=default_stem,
                                      label_visibility="collapsed",
                                      placeholder="filename (without .gcode)")
        with c_save:
            st.download_button(
                "Save G-code", text,
                file_name=(save_stem.strip() or default_stem) + ".gcode",
                mime="text/plain", icon=":material/save:",
                type="primary", use_container_width=True,
                disabled=(vrep.verdict == "fail"),
                help="Blocked while the file FAILs validation." if vrep.verdict == "fail" else None)

        # Shaded 3D preview of the actual G-code (base blue, walls clay).
        with st.spinner("Building toolpath preview…"):
            tp = extract_toolpath(text)
            groups = []
            for kind, color in (("base", BASE_CLR), ("wall", CLAY)):
                xs, ys, zs = tp[kind]
                groups.append((kind, split_gapped(xs, ys, zs), color, kind == "base"))
            fig = toolpath_figure(groups, file_lh, profile["bed_x"], profile["bed_y"],
                                  height=540, bead_width=nozzle)
        st.plotly_chart(fig, use_container_width=True, key="validate_preview")

        with st.expander("Preview first 60 lines"):
            st.code("\n".join(text.splitlines()[:60]), language="gcode")
    else:
        st.info("Upload a G-code file — or pick an included one — to validate.")
    render_footer()
    st.stop()


if shape_type == "Handle":
    st.caption("Live preview — the handle prints flat on the bed as one continuous "
               "bead. When it's leather-hard, score both ends and the pot, apply "
               "slip, and press it on.")
elif shape_type != "Virtual Wheel":
    st.caption("Live preview — shape it with the controls in the sidebar, then export below.")

# --- VIRTUAL WHEEL UI (MAIN COLUMN) ---
# Stage machine so everything happens in one spot (no scrolling):
#   draw -> the canvas;  spinning -> the spin animation replaces the canvas;
#   spun -> the Print Simulator replaces it.
if shape_type == "Virtual Wheel":
    vw_stage = st.session_state.get("vw_stage", "draw")
    if "vw_profile" not in st.session_state:
        st.session_state.vw_profile = None

    c_vw_title, c_vw_info = st.columns([6, 0.5])
    with c_vw_title:
        st.markdown(f"### {ui_icon('wheel', 22, ACC)} Virtual Wheel", unsafe_allow_html=True)
    with c_vw_info:
        with st.popover(":material/info:", help="How does this work?"):
            st.markdown(
                "**Draw, then spin.** Sketch the vessel's *profile* (its side "
                "silhouette) on the canvas. The **left edge is the axis of "
                "rotation** — like the center of a potter's wheel — and the "
                "**bottom line is the wheel base**: draw your wall down to it so "
                "the pot connects to its foot. Press **SPIN** and your sketch is "
                "revolved into a 3D vessel, then played back in the Print "
                "Simulator right here.")

    if vw_stage == "draw":
        # Web build: plain HTML5 canvas (the drawable-canvas custom component
        # doesn't run under stlite). On every stroke it postMessages the
        # sampled profile + a PNG snapshot to the host page, which writes
        # wheel_profile.json into the Pyodide filesystem (see index.html).
        components.html(r"""
<div style="font-family:'Segoe UI',sans-serif">
  <canvas id="pad" width="900" height="520"
    style="border:2px solid #E4DED7;border-radius:12px;background:#FBFAF8;touch-action:none;cursor:crosshair;max-width:100%"></canvas>
  <div style="margin-top:8px;display:flex;gap:10px">
    <button id="clear" style="padding:7px 16px;border:1px solid #E4DED7;border-radius:8px;background:#fff;cursor:pointer;font-weight:600">Clear</button>
    <span id="hint" style="align-self:center;color:#8A8178;font-size:0.9rem">
      Left edge = center axis · bottom line = wheel base. Draw the wall from the base up to the rim.</span>
  </div>
</div>
<script>
const cv=document.getElementById('pad'), ctx=cv.getContext('2d');
const W=cv.width, H=cv.height, AX=26, BASE=H-24;
let pts=[], drawing=false;
function grid(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#C99B7A'; ctx.setLineDash([7,7]); ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(AX,0); ctx.lineTo(AX,H); ctx.stroke();     // axis
  ctx.strokeStyle='#8FA3B8';
  ctx.beginPath(); ctx.moveTo(0,BASE); ctx.lineTo(W,BASE); ctx.stroke(); // wheel base
  ctx.setLineDash([]); ctx.fillStyle='#B9AEA4'; ctx.font='12px sans-serif';
  ctx.save(); ctx.translate(16,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('CENTER AXIS',-40,0); ctx.restore();
  ctx.fillText('WHEEL BASE — draw your wall down to this line', 40, BASE+16);
}
function redraw(){
  grid();
  if(pts.length>1){
    ctx.strokeStyle='#B0603C'; ctx.lineWidth=10; ctx.lineJoin='round'; ctx.lineCap='round';
    ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y);
    for(const p of pts) ctx.lineTo(p.x,p.y);
    ctx.stroke();
  }
}
function pos(e){
  const r=cv.getBoundingClientRect();
  const t=e.touches?e.touches[0]:e;
  const sx=cv.width/r.width, sy=cv.height/r.height;
  return {x:Math.max(AX,(t.clientX-r.left)*sx), y:Math.min(BASE,Math.max(0,(t.clientY-r.top)*sy))};
}
function start(e){drawing=true; pts=[pos(e)]; redraw(); e.preventDefault();}
function move(e){ if(!drawing)return; pts.push(pos(e)); redraw(); e.preventDefault();}
function end(e){ if(!drawing)return; drawing=false; send(); e.preventDefault();}
cv.addEventListener('pointerdown',start); cv.addEventListener('pointermove',move);
cv.addEventListener('pointerup',end); cv.addEventListener('pointerleave',end);
document.getElementById('clear').onclick=()=>{pts=[]; redraw();
  parent.postMessage({type:'clay_profile', profile:null}, '*');};
function build(){
  if(pts.length<5) return null;
  const N=72, bins=Array.from({length:N},()=>[]);
  for(const p of pts){
    let k=Math.round((1-(p.y/BASE))*(N-1)); k=Math.max(0,Math.min(N-1,k));
    bins[k].push(Math.max(0,p.x-AX));
  }
  let a=bins.map(b=> b.length? b.reduce((s,c)=>s+c,0)/b.length : null);
  let last=null; for(let i=0;i<N;i++){ if(a[i]==null)a[i]=last; else last=a[i]; }
  last=null; for(let i=N-1;i>=0;i--){ if(a[i]==null)a[i]=last; else last=a[i]; }
  if(a.some(v=>v==null)) return null;
  const rmax=Math.max(...a); if(rmax<=0) return null;
  return a.map(v=>Math.max(0,v/rmax));
}
function send(){
  const prof=build();
  parent.postMessage({type:'clay_profile', profile:prof,
                      png: prof? cv.toDataURL('image/png') : null}, '*');
  document.getElementById('hint').textContent = prof
    ? 'Looks good — press SPIN below.' : 'Draw one line from the base up to the rim.';
}
grid();
</script>""", height=600)

        if st.button("SPIN", type="primary", icon=":material/rotate_right:", use_container_width=True):
            _wp = {}
            if os.path.exists("wheel_profile.json"):
                try:
                    with open("wheel_profile.json") as _f:
                        _wp = json.load(_f)
                except Exception:
                    _wp = {}
            prof = _wp.get("profile")
            if prof and len(prof) >= 4:
                arr = np.clip(np.asarray(prof, dtype=float), 0.0, 1.5)
                _k = np.ones(5) / 5.0   # tame hand wobble -> clean, printable layers
                arr = np.convolve(np.pad(arr, 2, mode="edge"), _k, "valid")
                st.session_state.vw_profile = arr
                _png = (_wp.get("png") or "").split(",", 1)
                st.session_state.vw_sketch_b64 = _png[1] if len(_png) == 2 else ""
                st.session_state.vw_stage = "spinning"
                st.rerun()
            else:
                st.warning("Canvas is empty. Draw a wall from the WHEEL BASE line up, then press SPIN.")

        custom_profile_data = None
        st.stop()   # nothing below renders until the wheel has spun

    elif vw_stage == "spinning":
        # The animation takes the canvas's place.
        b64img = st.session_state.get("vw_sketch_b64", "")
        st.markdown(f"""
<style>@keyframes ezspin {{ from {{ transform: rotateY(0deg); }} to {{ transform: rotateY(360deg); }} }}</style>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:480px;">
  <img src="data:image/png;base64,{b64img}" style="height:340px;animation:ezspin 0.45s linear infinite;border-radius:10px;"/>
  <div style="margin-top:16px;font-weight:700;color:{ACC};letter-spacing:4px;font-size:1.1rem;">SPINNING…</div>
</div>""", unsafe_allow_html=True)
        time.sleep(1.5)
        # Hand the same spot over to the Print Simulator, already playing.
        st.session_state.vw_stage = "spun"
        st.session_state.sim_toggle = True
        st.session_state.sim_playing = True
        st.session_state.current_progress_idx = 0
        st.session_state.pop("scrubber", None)
        st.rerun()

    else:  # spun — simulator renders below, where the wheel was
        c_done, c_redraw = st.columns([4, 1.4])
        with c_done:
            st.markdown(
                f'<div class="ez-model-chip">{ui_icon("wheel", 18, ACC)} Spun from your drawing'
                f'<span class="ez-sub">watch it print below — or throw again</span></div>',
                unsafe_allow_html=True)
        with c_redraw:
            if st.button("Back to the wheel", icon=":material/undo:", use_container_width=True):
                st.session_state.vw_stage = "draw"
                st.session_state.sim_playing = False
                st.rerun()
        custom_profile_data = st.session_state.vw_profile
        if custom_profile_data is None:
            st.session_state.vw_stage = "draw"
            st.rerun()


# --- GENERATION ---


# --- GENERATION ---
prof_func = None
if shape_type == "Cylinder": prof_func = profile_cylinder
elif shape_type == "Bowl": prof_func = profile_bowl
elif shape_type == "Vase": prof_func = profile_vase
elif shape_type == "Virtual Wheel" and custom_profile_data is not None:
    # Interpolate from the custom profile array
    def profile_custom(t, r_max):
        # t is 0..1
        # Map t to index
        idx = int(t * (len(custom_profile_data) - 1))
        idx = max(0, min(idx, len(custom_profile_data) - 1))
        norm_r = custom_profile_data[idx]
        return norm_r * r_max
        
    prof_func = profile_custom

text_func = None
if tex_type == "Sine Waves":
    text_func = lambda t, a: texture_sine_waves(t, a, freq=tex_freq, amp=tex_amp)
elif tex_type == "Twist":
    text_func = lambda t, a: texture_twist(t, a, twist_factor=twist_factor, ridges=int(tex_freq), amp=tex_amp)
elif tex_type == "Image Map" and img_array is not None:
    text_func = lambda t, a: texture_from_image(t, a, img_array=img_array, amp=tex_amp)

if shape_type == "Handle":
    # Flat-printed handle(s): curve on the bed, strap width built up in Z.
    # Spacing is bed-aware so multiple copies stay inside the print area
    # (footprint grows with extra beads).
    _beads = int(hd_beads)
    _w_eff = handle_w + (_beads - 1) * nozzle
    _spacing = _w_eff + 16.0
    if int(handle_copies) > 1:
        _spacing = min(_spacing,
                       (profile["bed_x"] - 12.0 - _w_eff) / (int(handle_copies) - 1))
        if _spacing < _w_eff + nozzle + 2:
            st.warning(f"{int(handle_copies)} copies of a {handle_w:.0f} mm handle "
                       "are a squeeze on this bed — the copies print very close "
                       "together. Fewer copies or a narrower handle is safer.")
    clay_obj = generate_handle_path(
        style=handle_style, width=handle_w, height=handle_h,
        strap_width=strap_w, layer_height=layer_h, nozzle_diameter=nozzle,
        thickness_beads=_beads,
        copies=int(handle_copies), spacing=_spacing,
        first_layer_height=first_layer_h,
        kuksa_d=0.55 + 0.449 * (kuksa_dip / 100.0),
    )
else:
    clay_obj = generate_spiral_path(
        body_base_radius=body_base_radius,
        total_height=height,
        layer_height=layer_h,
        nozzle_diameter=nozzle,
        base_height=base_h,
        base_bottom_radius=base_bottom_radius,
        sides=roundness,
        profile_func=prof_func,
        texture_func=text_func,
        first_layer_height=first_layer_h,
    )

# --- VISUALIZATION ---
pts = clay_obj.to_numpy()

if len(pts) > 0:
    # Preview Mode. Shaded is the default view; the Print Simulator is an
    # opt-in extra on the right (experimental — it takes over the preview
    # while enabled, so the two never render together).
    c_shaded, c_sim = st.columns([2, 1.4])
    with c_sim:
        show_preview = st.toggle(
            "Print Simulator (experimental)", key="sim_toggle",
            help="Watch the print run line by line with a play/scrub bar. "
                 "Experimental — takes over the preview while on.")
    with c_shaded:
        design_shaded = st.toggle(
            "Shaded preview", value=True, key="design_shaded",
            disabled=show_preview,
            help="Render the vessel as solid clay beads, like the Slice tab. "
                 "Heavier on the browser — turn off if it feels sluggish.")
    # The simulator replaces the shaded view while it is on.
    design_shaded = design_shaded and not show_preview

    if design_shaded:
        # Same lit bead-mesh renderer the Slice tab uses. Offset the origin-
        # centered design onto the bed, split base vs wall at the base height.
        cxo, cyo = profile["center_x"], profile["center_y"]
        _travels = sorted(getattr(clay_obj, "travel_idx", set()))
        if _travels:
            # Handles with copies: split at travel hops so the preview doesn't
            # draw a clay tube between separate pieces.
            d_groups = []
            _runs, _prev = [], 0
            for _ti in _travels:
                _runs.append(pts[_prev:_ti]); _prev = _ti
            _runs.append(pts[_prev:])
            d_groups.append(("handle",
                             [(r[:, 0] + cxo, r[:, 1] + cyo, r[:, 2])
                              for r in _runs if len(r) > 1], CLAY, False))
        else:
            above = pts[:, 2] > base_h + 1e-6
            split = int(np.argmax(above)) if above.any() else len(pts)
            d_groups = []
            bpts, wpts = pts[:split], pts[split:]
            if len(bpts) > 1:
                d_groups.append(("base", [(bpts[:, 0] + cxo, bpts[:, 1] + cyo, bpts[:, 2])],
                                 BASE_CLR, False))
            if len(wpts) > 1:
                d_groups.append(("wall", [(wpts[:, 0] + cxo, wpts[:, 1] + cyo, wpts[:, 2])],
                                 CLAY, False))
        st.plotly_chart(
            toolpath_figure(d_groups, layer_h, profile["bed_x"], profile["bed_y"],
                            height=640, bead_width=nozzle),
            use_container_width=True, key="design_shaded_preview")

    # Defaults
    plot_pts = pts
    nozzle_pos = None
    squishy_clay_look = False
    show_layer_lines = False
    show_nozzle = False

    # Preview Controls
    if show_preview:
        col_play, col_progress, col_opts = st.columns([0.5, 3.5, 2])

        if "current_progress_idx" not in st.session_state:
            st.session_state.current_progress_idx = 0

        playing = st.session_state.get("sim_playing", False)

        with col_play:
            if st.button(":material/pause:" if playing else ":material/play_arrow:",
                         key="sim_play_btn",
                         help="Watch the print run on its own"):
                playing = not playing
                st.session_state.sim_playing = playing
                if playing and st.session_state.current_progress_idx >= len(pts):
                    st.session_state.current_progress_idx = 0  # replay from start
                st.session_state.pop("scrubber", None)
                st.rerun()

        # While playing, advance the scrubber each run (the widget key is reset
        # so the slider follows the animated position).
        if playing:
            step = max(len(pts) // 140, 10)
            nxt = st.session_state.current_progress_idx + step
            if nxt >= len(pts):
                nxt = len(pts)
                st.session_state.sim_playing = False
            st.session_state.current_progress_idx = nxt
            st.session_state.pop("scrubber", None)

        with col_progress:
            current_progress_idx = st.slider(
                "Print Progress", 0, len(pts), st.session_state.current_progress_idx,
                step=10, key="scrubber", label_visibility="collapsed")
            if not playing:
                st.session_state.current_progress_idx = current_progress_idx

        with col_opts:
            c_clay, c_nozz = st.columns(2)
            with c_clay:
                squishy_clay_look = st.checkbox("Clay View", value=True)
                show_layer_lines = st.checkbox("Show Layer Lines", value=True)
            with c_nozz:
                show_nozzle = st.checkbox("Show Nozzle", value=True)

        plot_pts = pts[:st.session_state.current_progress_idx]
        if st.session_state.current_progress_idx < len(pts) and st.session_state.current_progress_idx > 0:
            nozzle_pos = pts[st.session_state.current_progress_idx-1]
        elif st.session_state.current_progress_idx == 0:
            if len(pts) > 0: nozzle_pos = pts[0]

    # Style Logic
    line_width = max(2, nozzle * 2)
    line_color = plot_pts[:, 2] if len(plot_pts) > 0 else "white"
    line_colorscale = 'Plasma'
    
    traces = []
    
    if show_preview and squishy_clay_look:
        # Clay View = the SAME lit bead-tube renderer as the shaded/Slice
        # previews. The full tube mesh is precomputed once and cached in the
        # session (keyed by path + settings); each animation frame is then two
        # numpy slices up to the print-progress point — no geometry rebuild,
        # and early frames ship only the clay printed so far.
        _stride = 1 + len(pts) // 12000   # cap mesh size so frames stay light
        _sim_key = (hashlib.md5(pts.tobytes()).hexdigest()
                    + f":{layer_h}:{nozzle}:{CLAY}:{_stride}")
        _cached = st.session_state.get("sim_mesh_cache")
        if not _cached or _cached[0] != _sim_key:
            _spts = pts[::_stride]
            _cached = (_sim_key,
                       bead_mesh_arrays(_spts[:, 0], _spts[:, 1], _spts[:, 2],
                                        layer_h, CLAY, bead_width=nozzle))
            st.session_state.sim_mesh_cache = _cached

        _upto = -(-st.session_state.current_progress_idx // _stride)  # ceil
        _mesh = partial_bead_mesh(_cached[1], upto=_upto) if _cached[1] else None
        if _mesh is not None:
            traces.append(_mesh)

        # Layer Lines: thin dark seam line along the bead tops (thick black
        # was for the old fat-line body; over a real mesh it would bury it).
        if show_layer_lines and len(plot_pts) > 1:
            lp = plot_pts[::_stride]
            traces.append(go.Scatter3d(
                x=lp[:, 0], y=lp[:, 1], z=lp[:, 2],
                mode='lines',
                name='Layer Lines',
                line=dict(color='rgba(0,0,0,0.55)', width=2),
                hoverinfo='skip'
            ))

    else:
        # Standard View (Heatmap)
        traces.append(go.Scatter3d(
            x=plot_pts[:, 0],
            y=plot_pts[:, 1],
            z=plot_pts[:, 2],
            mode='lines',
            name='Path',
            line=dict(
                color=line_color, 
                colorscale=line_colorscale,
                cmin=np.min(pts[:, 2]),
                cmax=np.max(pts[:, 2]),
                width=line_width
            )
        ))
    
    # Nozzle (if previewing)
    if show_preview and show_nozzle and nozzle_pos is not None:
        traces.append(go.Scatter3d(
            x=[nozzle_pos[0]],
            y=[nozzle_pos[1]],
            z=[nozzle_pos[2]],
            mode='markers',
            name='Nozzle',
            marker=dict(
                color='#FFD700', # Gold
                size=8,
                symbol='diamond',
                line=dict(width=1, color='black')
            )
        ))
        
        # Ghost Path (strided — it's a faint guide, full resolution is waste)
        if len(plot_pts) < len(pts):
            ghost_pts = pts[st.session_state.current_progress_idx::max(1, len(pts) // 12000)]
            traces.append(go.Scatter3d(
                x=ghost_pts[:, 0],
                y=ghost_pts[:, 1],
                z=ghost_pts[:, 2],
                mode='lines',
                name='Future Path',
                line=dict(color='lightgrey', width=2),
                opacity=0.2
            ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        # Keep the user's camera angle across animation frames/reruns —
        # without this every frame snapped the view back to the default.
        uirevision="design_sim",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=True, title="Height (mm)"),
            aspectmode='data'
        ),
        legend=dict(
            y=0.8,
            x=1.0,
            xanchor='right',
            yanchor='top'
        ),
        height=700,
        margin=dict(r=0, l=0, b=0, t=0)
    )

    if not design_shaded:
        st.plotly_chart(fig, use_container_width=True)

    # Autoplay: while the simulator is playing, advance a frame and rerun.
    if show_preview and st.session_state.get("sim_playing"):
        time.sleep(0.12)
        st.rerun()
else:
    st.warning("No path generated.")

# --- EXPORT ---
st.divider()

# Center the toolpath on the machine's bed center (Eazao is cartesian).
offset_x = profile["center_x"]
offset_y = profile["center_y"]

# Descriptive provenance: filename + ;SOURCE: header say exactly what this is,
# so a parametric design can't be mistaken for a sliced STL.
_shape_slug = shape_type.lower().replace(" ", "")
design_name = f"design_{_shape_slug}_r{body_base_radius:g}_h{height:g}"
gcode_str = generate_gcode(
    clay_obj, offset_x=offset_x, offset_y=offset_y, profile=profile,
    line_width=nozzle, first_layer_flow=first_layer_flow,
    source=f"Design (parametric {shape_type}, max radius {body_base_radius:g} mm, "
           f"height {height:g} mm) on {printer_name}")

# Auto-validate the design against the printer before offering the download.
st.markdown('<div class="ez-step"><span class="ez-stepnum">✓</span> Validate & export '
            '<span class="ez-stepsub">checked automatically before every download</span></div>',
            unsafe_allow_html=True)
render_validation(gcode_str, profile, nozzle=nozzle, layer_height=layer_h)

c_ex1, c_ex2 = st.columns(2)

with c_ex1:
    st.download_button("Download G-Code", gcode_str, file_name=design_name + ".gcode",
                       icon=":material/download:", mime="text/plain", use_container_width=True)

with c_ex2:
    # Generate STL on demand; the result lives in session so the download
    # button survives reruns (e.g. simulator playback).
    if shape_type == "Handle":
        st.caption("Handles export as G-code (the STL mesher is built for "
                   "revolved vessels).")
    elif st.button("Generate STL Model", use_container_width=True):
        with st.spinner("Meshing STL..."):
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                try:
                    success = generate_stl_from_path(clay_obj, tmp.name)
                except ImportError:
                    # numpy-stl isn't bundled in the web build.
                    success = False
                    st.info("STL export ships with the desktop build — the "
                            "G-code download above is everything the printer needs.")
                tmp_path = tmp.name
            if success:
                with open(tmp_path, "rb") as f:
                    st.session_state.design_stl = f.read()
                os.unlink(tmp_path)
            else:
                st.session_state.design_stl = None
                st.error("STL Generation Failed (numpy-stl missing or path empty).")
    if st.session_state.get("design_stl"):
        st.download_button("Download STL", st.session_state.design_stl,
                           file_name=design_name + ".stl", mime="model/stl",
                           icon=":material/download:", use_container_width=True)

render_footer()
