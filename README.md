# ClayShaper

**A ceramic slicer for clay 3D printers — free, and it runs in your browser.**

ClayShaper turns models into printable clay toolpaths: continuous beads, solid
staggered bases that actually bond, fold softening for tricky shapes, and a
validator that catches bad files *before* they reach the machine. It is tuned
for the **Eazao Potter** (165 × 165 × 280 mm) and you can add your own printer.

Everything runs on your own machine. Models never leave your computer.

---

## What it does

- **Slice STL** — vase (single continuous spiral) or wall mode, staggered base
  with inset fill, fold softening, scaling, and a lit 3D bead preview.
- **Design** — parametric cylinders / bowls / vases with textures.
- **Virtual Wheel** — draw a profile, spin it into a pot, watch it print.
- **Handles** — 9 styles (half circle, 3/4 circle, kuksa, heart, hook…), 1–6
  beads thick, multiple copies, printed flat and attached when leather-hard.
- **Validate G-code** — bounds, feedrates, the mandatory clay start codes
  (`M302` / `M163` / `M164`), over-extrusion, walls printing into thin air,
  and clay usage in grams. Four verdicts: pass, pass-with-tips, caution, fail.

## Run it

**Desktop (Windows)** — grab `ClayShaper.exe` from
[Releases](../../releases), double-click, done. ~10 MB, no install, no Python.

**From source** — it's a static site; any file server works:

```bash
python -m http.server 8513 --directory web
```

then open <http://localhost:8513>.

**Build the .exe:**

```bash
pip install pyinstaller && cd desktop && pyinstaller ClayShaper.spec
```

## How it works

The whole app is Python compiled to WebAssembly
([stlite](https://github.com/whitphx/stlite) / Pyodide), so the slicing engine
runs inside the browser tab — no server, no uploads. The desktop build is the
same site wrapped in a tiny local file server.

```
web/        the app + slicing engine (this is the whole product)
  app.py            UI
  stl_slicer.py     STL -> clay toolpaths
  clay_lib.py       parametric shapes, handles, G-code writer
  gcode_validator.py  the safety net
  preview3d.py      lit bead-mesh preview
  nearest.py        small numpy stand-ins for scipy bits
desktop/    PyInstaller launcher that serves web/ and opens a browser
```

### Sample models are not included

The models and factory G-code I test against belong to Eazao and other
designers, so they aren't redistributed here. Drop your own `.stl` files into
`web/assets/models/` and they appear in the gallery automatically. Uploading
your own STL in the app always works without any of this.

## Notes for clay

Clay can't retract, so every path is continuous. Line width equals the nozzle
bore, extrusion is volumetric, and the start/end blocks match what the Potter
expects. If a slice can only be partly read from a mesh, the app tells you
where it stopped instead of quietly handing you a stump.

## Support it

ClayShaper is free. Comparable slicers run $200–300 a year. If it saved you
something, [buy me a coffee](https://buymeacoffee.com/randotechnerd) — any
donation gets you the desktop build and a year of updates, and it's what keeps
this going.

## License

Free to download, use, and modify — including commercially (sell the pots!).
**Redistribution needs written permission**: bundling with a printer, shipping
it on media, or handing out copies in volume. See [LICENSE.md](LICENSE.md).
For makers and small studios the answer is usually yes — just ask.

---

RandoTechNerd@gmail.com · [YouTube](https://www.youtube.com/@RandoTechNerd) ·
[RandoTechNerd.com](https://randotechnerd.com)
