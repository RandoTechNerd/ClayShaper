"""
ClayShaper branding: colors, the wordmark, and simple line-drawn printer
icons for the machine picker.

Design language taken from eazao.com: black lowercase rounded wordmark, a
slate-blue + terracotta accent palette, clean white surfaces with warm-grey
cards, and uppercase letter-spaced section labels.
"""

# --- Palette ---------------------------------------------------------------
INK = "#1A1A1A"          # near-black, wordmark + headings
SLATE = "#3D4A5C"        # slate blue (their nav "STORE" pill)
TERRACOTTA = "#B0603C"   # clay accent (their buttons)
TERRACOTTA_DK = "#8A4A2D"
CREAM = "#F7F5F2"        # warm off-white card surface
LINE = "#E4DED7"         # hairline borders
MUTED = "#8A8178"        # secondary text

# Selectable themes (Advanced menu). Each restyles the whole app: accent,
# sidebar, app background and text colors.
THEMES = {
    "Terracotta": {
        "acc": TERRACOTTA, "acc_dk": TERRACOTTA_DK,
        "side": "#F7F5F2", "app": "#FFFFFF",
        "ink": "#1A1A1A", "muted": "#8A8178", "line": "#E4DED7",
    },
    "Slate": {
        "acc": SLATE, "acc_dk": "#2B3542",
        "side": "#EDF0F4", "app": "#FAFBFC",
        "ink": "#141A22", "muted": "#6E7A88", "line": "#D8DEE6",
    },
    "Kiln Red": {
        "acc": "#A62B1F", "acc_dk": "#7E2015",
        "side": "#F8EFEA", "app": "#FFFCFA",
        "ink": "#241310", "muted": "#93756B", "line": "#EAD9D0",
    },
    "Celadon": {
        "acc": "#5E8B6F", "acc_dk": "#456B54",
        "side": "#EDF3EE", "app": "#FBFCFA",
        "ink": "#15201A", "muted": "#75857A", "line": "#D9E3DA",
    },
}

# Back-compat alias (accent pairs only).
ACCENTS = {k: (v["acc"], v["acc_dk"]) for k, v in THEMES.items()}


def ui_icon(name, size=20, color=None):
    """Small inline stroke icons in the same language as the printer drawings."""
    c = color or TERRACOTTA
    body = {
        "slice": '<path d="M4 17 L12 21 L20 17 M4 12 L12 16 L20 12 M4 7 L12 11 L20 7 L12 3 Z"/>',
        "design": '<path d="M12 21 C6 21 5 15 7 10 C9 5 15 5 17 10 C19 15 18 21 12 21 Z M12 3 v4"/>',
        "validate": '<path d="M12 3 L20 6 V12 C20 17 16 20 12 21 C8 20 4 17 4 12 V6 Z"/><path d="M9 12 l2.2 2.2 L15.5 9.5"/>',
        "gear": '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v3 M12 18.2v3 M2.8 12h3 M18.2 12h3 M5.5 5.5l2.1 2.1 M16.4 16.4l2.1 2.1 M18.5 5.5l-2.1 2.1 M7.6 16.4l-2.1 2.1"/>',
        "wheel": '<circle cx="12" cy="13" r="7"/><circle cx="12" cy="13" r="2.4"/><path d="M4 21 h16"/>',
    }.get(name, "")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{c}" stroke-width="1.9" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:-3px">{body}</svg>')


def wordmark_svg(height=42):
    """The 'ClayShaper' wordmark: 'Clay' in ink, 'Shaper' in terracotta, one
    continuous lockup. viewBox is generous so nothing ever clips."""
    return f"""
<svg width="{height*5.4:.0f}" height="{height}" viewBox="0 0 540 100"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ClayShaper">
  <text x="0" y="72" font-family="'Poppins','Segoe UI',sans-serif"
        font-size="76" font-weight="700" letter-spacing="-2"><tspan
        fill="{INK}">Clay</tspan><tspan fill="{TERRACOTTA}">Shaper</tspan></text>
</svg>"""


# --- Printer catalog -------------------------------------------------------
# Each icon is a tiny, consistent line drawing (~90x80 viewBox) evoking the real
# machine silhouette. Only "Eazao Potter" is wired to slice for now; the rest
# are shown as selectable-but-disabled "coming soon" options.

_S = TERRACOTTA          # stroke for the active nozzle/clay accent


def _potter_svg(stroke):
    # Cantilever arm + vertical column + wide flat base (the Potter/Zero form).
    return f"""
<svg viewBox="0 0 90 80" xmlns="http://www.w3.org/2000/svg" fill="none"
     stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <rect x="52" y="10" width="14" height="46" rx="2"/>
  <path d="M52 20 H20"/>
  <path d="M20 20 V30"/>
  <circle cx="20" cy="34" r="3" fill="{stroke}" stroke="none"/>
  <path d="M14 64 H80 L74 56 H20 Z"/>
  <path d="M66 24 h10"/>
</svg>"""


def _zero_svg(stroke):
    # Like the Potter but with a spool/feed loop on top (Zero silhouette).
    return f"""
<svg viewBox="0 0 90 80" xmlns="http://www.w3.org/2000/svg" fill="none"
     stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <rect x="54" y="12" width="13" height="44" rx="2"/>
  <path d="M54 22 H22"/>
  <circle cx="22" cy="26" r="3" fill="{stroke}" stroke="none"/>
  <path d="M60 12 C60 4 78 4 78 14 C78 20 70 20 70 20"/>
  <path d="M16 64 H80 L74 56 H22 Z"/>
</svg>"""


def _matrix_svg(stroke):
    # Enclosed cube-frame gantry (Matrix M-series silhouette).
    return f"""
<svg viewBox="0 0 90 80" xmlns="http://www.w3.org/2000/svg" fill="none"
     stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <rect x="16" y="12" width="58" height="54" rx="2"/>
  <path d="M16 24 H74"/>
  <path d="M45 24 V34"/>
  <circle cx="45" cy="37" r="3" fill="{stroke}" stroke="none"/>
  <path d="M34 58 h22"/>
</svg>"""


def _tong_svg(stroke):
    # Wide open-frame concrete gantry (Tong A-series): broad base, overhead beam.
    return f"""
<svg viewBox="0 0 90 80" xmlns="http://www.w3.org/2000/svg" fill="none"
     stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <path d="M10 66 V22 H80 V66"/>
  <path d="M10 22 H80"/>
  <path d="M45 22 V32"/>
  <circle cx="45" cy="35" r="3" fill="{stroke}" stroke="none"/>
  <path d="M6 66 H84"/>
</svg>"""


# name -> (svg_fn, tagline, active)
PRINTER_CATALOG = [
    ("Eazao Potter", _potter_svg, "Desktop clay · 165×165×280", True),
    ("Eazao Zero",   _zero_svg,   "Compact clay", False),
    ("Matrix M500",  _matrix_svg, "Enclosed multi-material", False),
    ("Tong A1000",   _tong_svg,   "Large-format concrete", False),
]


def printer_icon(name, active=True, stroke=None):
    """Return the SVG string for a printer by name."""
    fn = dict((n, f) for n, f, _, _ in PRINTER_CATALOG).get(name, _potter_svg)
    return fn(stroke or (TERRACOTTA if active else "#C9C2BA"))
