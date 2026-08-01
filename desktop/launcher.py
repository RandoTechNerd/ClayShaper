"""
ClayShaper desktop launcher.

The whole app is the browser build (Python-in-WebAssembly): this launcher just
serves the bundled site from a local port and opens the default browser. That
keeps the download tiny — there's no Python runtime, no Streamlit server and no
install; the slicing happens inside the browser tab, on the user's own machine.

Build:  pyinstaller ClayShaper.spec
"""
import http.server
import os
import socket
import socketserver
import sys
import threading
import time
import webbrowser

APP_NAME = "ClayShaper"
PREFERRED_PORTS = (8770, 8771, 8772, 8773, 8774, 0)


def site_dir():
    """The bundled static site (PyInstaller unpacks data to _MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "site")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=site_dir(), **kw)

    def end_headers(self):
        # The page pulls its Python runtime once and relies on the browser
        # cache afterwards; never let our own files go stale between versions.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass   # keep the console clean


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def find_port():
    for port in PREFERRED_PORTS:
        try:
            srv = Server(("127.0.0.1", port), Handler)
            return srv, srv.server_address[1]
        except OSError:
            continue
    raise SystemExit("Could not open a local port for ClayShaper.")


def main():
    if not os.path.isdir(site_dir()):
        print("ERROR: the bundled site folder is missing.")
        input("Press Enter to close...")
        return

    srv, port = find_port()
    url = f"http://127.0.0.1:{port}/"

    print("=" * 62)
    print(f"  {APP_NAME}  -  ceramic slicer")
    print("=" * 62)
    print()
    print(f"  Opening {url}")
    print()
    print("  The first run downloads the engine into your browser cache")
    print("  (about 30 seconds, needs internet once). After that it starts")
    print("  fast and your models never leave this computer.")
    print()
    print("  Keep this window open while you work.")
    print("  Close it (or press Ctrl+C) to quit ClayShaper.")
    print()
    print("=" * 62)

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)
    try:
        webbrowser.open(url)
    except Exception:
        print(f"  Could not open a browser automatically — visit {url}")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        print("\nClayShaper stopped.")


if __name__ == "__main__":
    main()
