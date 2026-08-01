"""
ClayShaper desktop launcher.

The app itself is the browser build (Python compiled to WebAssembly). This
launcher serves it from a local port and opens it in an "app window" — a
Chrome/Edge window with no address bar, no tabs and its own icon — so it
looks and behaves like a normal desktop program rather than a web page.

Design notes:
  * Its own browser profile lives in %LOCALAPPDATA%\\ClayShaper\\browser so the
    WebAssembly engine is cached between launches (first run downloads it,
    every run after that starts in a couple of seconds).
  * The launcher waits on the browser window: close the window and the server
    shuts down and the process exits. No stray console to tidy up.
  * If no Chromium-based browser is found it falls back to the default
    browser and keeps running until the user closes the console.

Build:  pyinstaller ClayShaper.spec
"""
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

APP_NAME = "ClayShaper"
PREFERRED_PORTS = (8770, 8771, 8772, 8773, 8774, 0)
WINDOW = "--window-size=1500,960"


def site_dir():
    """The bundled static site (PyInstaller unpacks data to _MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "site")


def profile_dir():
    """Persistent browser profile — keeps the engine cached between runs."""
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(root, APP_NAME, "browser")
    os.makedirs(path, exist_ok=True)
    return path


def find_browser():
    """First Chromium-based browser we can drive in app mode."""
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def fatal(message):
    """Show a real dialog — there's no console window to print to."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, f"{APP_NAME}", 0x10)
    except Exception:
        print(message)
    sys.exit(1)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=site_dir(), **kw)

    def end_headers(self):
        # The page caches its engine in the browser; never serve stale app files.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server():
    for port in PREFERRED_PORTS:
        try:
            srv = Server(("127.0.0.1", port), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            return srv, srv.server_address[1]
        except OSError:
            continue
    fatal("ClayShaper could not open a local port.\n\n"
          "Something may already be using ports 8770-8774.")


def main():
    if not os.path.isdir(site_dir()):
        fatal("ClayShaper is missing its program files.\n\n"
              "Try extracting the download again, then run ClayShaper.exe.")

    srv, port = start_server()
    url = f"http://127.0.0.1:{port}/"
    time.sleep(0.3)

    browser = find_browser()
    if browser:
        # App mode: no address bar, no tab strip — a plain application window.
        proc = subprocess.Popen([
            browser,
            f"--app={url}",
            f"--user-data-dir={profile_dir()}",
            WINDOW,
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=Translate,AutofillServerCommunication",
        ])
        try:
            proc.wait()          # closing the window ends the app
        except KeyboardInterrupt:
            pass
        finally:
            srv.shutdown()
        return

    # No Chromium browser: fall back to whatever the system uses.
    webbrowser.open(url)
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            f"{APP_NAME} is running in your browser.\n\n"
            f"If it didn't open, go to:  {url}\n\n"
            "Click OK to close ClayShaper when you're finished.",
            APP_NAME, 0x40)
    except Exception:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    srv.shutdown()


if __name__ == "__main__":
    main()
