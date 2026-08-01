# PyInstaller spec — ClayShaper desktop launcher.
# The payload is the static browser build in site/; the executable itself is
# just a local file server + browser opener, so the download stays tiny.
# Build with:  pyinstaller ClayShaper.spec

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('site', 'site')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Nothing scientific runs natively here — the engine lives in the browser.
    # NOTE: do not exclude stdlib http.server needs (email, xml, asyncio…);
    # only drop the heavy third-party stacks and unused UI/test packages.
    excludes=['numpy', 'scipy', 'shapely', 'trimesh', 'plotly', 'streamlit',
              'PIL', 'pandas', 'matplotlib', 'tkinter', 'test', 'unittest',
              'pydoc_data'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ClayShaper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # the console window is the "keep me open" control
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ClayShaper.ico',
)
