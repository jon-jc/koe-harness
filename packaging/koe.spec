# PyInstaller spec for the koe desktop application.
#
# Run through `packaging/build.py`, which sets the environment this reads and
# then verifies the result. Building this spec directly works but skips the
# post-build checks.
#
# Two decisions worth knowing about:
#
#   * **Japanese support is a build flag.** unidic-lite is 248 MB — the single
#     largest contributor to the artifact — and koe degrades honestly without
#     it, falling back to character segmentation and *labelling* that fallback
#     wherever a number depends on it. It is included by default anyway,
#     because Japanese is the product's premise and a slim build would be the
#     wrong default for its audience. `KOE_BUILD_JAPANESE=0` produces the
#     smaller one.
#
#   * **One directory, not one file.** A `--onefile` build extracts ~400 MB to
#     a temporary directory on every launch, which costs several seconds of
#     startup and confuses antivirus. The installer hides the directory from
#     the user, which is what `--onefile` was really being used to achieve.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
WITH_JAPANESE = os.environ.get("KOE_BUILD_JAPANESE", "1") != "0"

datas = []
binaries = []
hiddenimports = []

# The web client. Resolved at runtime through koe.desktop.paths.resource_root(),
# which knows the difference between a checkout and a bundle.
datas += [
    (str(ROOT / "web" / "index.html"), "web"),
    (str(ROOT / "web" / "dist"), "web/dist"),
]

# The bundled evaluation corpus, so `Play demo` works offline.
if (ROOT / "datasets").is_dir():
    datas += [(str(ROOT / "datasets"), "datasets")]

# uvicorn resolves its loop and protocol implementations by string at runtime,
# so static analysis cannot see them.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
]

# pywebview loads a platform backend dynamically and ships JS it injects into
# the page; collect_all picks up both.
webview_datas, webview_binaries, webview_hidden = collect_all("webview")
datas += webview_datas
binaries += webview_binaries
hiddenimports += webview_hidden

# pydantic v2 has compiled internals that the analyser under-reports.
hiddenimports += collect_submodules("pydantic")
datas += collect_data_files("anthropic", include_py_files=False)

if WITH_JAPANESE:
    # fugashi carries a compiled extension and links libmecab; unidic_lite is
    # pure data. Both have to be collected explicitly.
    fugashi_datas, fugashi_binaries, fugashi_hidden = collect_all("fugashi")
    datas += fugashi_datas
    binaries += fugashi_binaries
    hiddenimports += fugashi_hidden
    datas += collect_data_files("unidic_lite")
    hiddenimports += ["fugashi", "unidic_lite"]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Excluded deliberately: pulled in transitively, never imported by koe, and
    # collectively worth well over 100 MB.
    excludes=[
        "tkinter",
        "matplotlib",
        "numpy.testing",
        "pytest",
        "IPython",
        "notebook",
        "PIL",
        "setuptools",
        "pip",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="koe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed binaries are a reliable antivirus false positive
    console=False,  # a GUI app; a console window flashing up looks broken
    icon=str(ROOT / "packaging" / "assets" / "koe.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="koe",
)
