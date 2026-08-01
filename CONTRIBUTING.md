# Contributing

Contributions are genuinely welcome — especially from people who actually
print clay. Bug reports from real failed prints are the most valuable thing
you can send.

## Good first things to send

- **A print that failed** — attach the `.stl` (if you can share it), your
  settings, and a photo. Half the fixes in this slicer came from exactly that.
- **A printer profile** — bed size, nozzle range, speed limits, and the
  start/end G-code your machine needs.
- **Sample models you own** and are happy to have shipped with the app.

## Ground rules

1. **Open an issue first** for anything bigger than a small fix, so we don't
   both build the same thing.
2. **Keep the clay rules intact.** Clay can't retract: paths must stay
   continuous, line width equals the nozzle bore, and extrusion is volumetric.
   A change that introduces travel moves mid-wall will be rejected.
3. **Test against real files.** If you touch the slicer or validator, check
   your change against Eazao's factory G-code and a few sample models — the
   validator should still pass all of them.
4. **No new heavy dependencies.** The app runs in WebAssembly; every package
   is a download the user waits through. `scipy` alone is ~30 MB. Optional
   C-extension packages (`rtree`, etc.) may not exist there at all.

## How merging works

This is a source-available project, not a community-governed one: I review
everything and decide what lands in the main app. Forks are fine — but note
that redistributing the software (yours or mine) needs written permission,
see [LICENSE.md](LICENSE.md).

By submitting a contribution you agree that it can be included in ClayShaper
and distributed under the project's license.

## Getting set up

```bash
python -m http.server 8513 --directory web   # then open localhost:8513
```

That's it — no build step. Edit the files in `web/` and reload.

Questions: RandoTechNerd@gmail.com
