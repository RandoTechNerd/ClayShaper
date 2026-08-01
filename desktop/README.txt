ClayShaper — ceramic slicer
===========================

TO RUN
------
Double-click  ClayShaper.exe

A small black window opens and your browser launches with ClayShaper.
Keep that window open while you work; close it to quit.

Windows SmartScreen may warn about an unrecognised app (the .exe isn't
code-signed yet). Click "More info" -> "Run anyway".

FIRST RUN
---------
The first launch downloads the engine into your browser cache — about
20-30 seconds, and it needs internet that one time. After that it starts
in a couple of seconds, and your models never leave your computer.

WHAT'S INSIDE
-------------
* Slice STL      - drop in your own STL, or use the 4 included samples
                   (Bowl, Spiral Vase, Octopus Vase, Taco Bell Bag)
* Design         - parametric cylinder / bowl / vase, textures,
                   the Virtual Wheel (draw a profile, spin it),
                   and Handles (9 styles, 1-6 beads thick, copies)
* Validate G-code- check any file before it hits the machine

Everything is tuned for the Eazao Potter (165 x 165 x 280 mm).

TROUBLESHOOTING
---------------
* Browser didn't open? Go to  http://127.0.0.1:8770
* Port busy? The app tries 8770-8774 automatically; the console window
  prints the address it actually used.
* Antivirus quarantine? PyInstaller apps are sometimes false-positived.
  The source is open — see the repo.

RandoTechNerd@gmail.com  ·  youtube.com/@RandoTechNerd  ·  RandoTechNerd.com
