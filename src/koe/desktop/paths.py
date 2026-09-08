"""Filesystem locations for the desktop application.

Two problems this solves, both of which only appear once the app is frozen and
therefore only appear after you thought you were finished.

**Resources move.** In a source checkout the web client sits next to the
package; inside a PyInstaller bundle it sits in a temporary extraction
directory named by ``sys._MEIPASS``. Code that hardcodes the first path works
perfectly until the day it is packaged.

**Writable locations are not where the app lives.** A packaged app may live in
``C:\\Program Files``, which is read-only for a standard user. Logs, settings
and recordings have to go to per-user directories, and those differ per
platform. Writing next to the executable is the mistake that turns into "works
on my machine, silently fails to save anything for everyone else".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "koe"


def is_frozen() -> bool:
    """Whether this is running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Directory containing bundled read-only resources.

    Frozen: the PyInstaller extraction directory. Source: the repository root,
    four levels up from this file (``src/koe/desktop/paths.py``).
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[3]


def web_root() -> Path:
    """Where ``index.html`` and ``dist/`` live."""
    return resource_root() / "web"


def _base_dir(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value) if value else fallback


def config_dir() -> Path:
    """Per-user configuration directory.

    Windows: ``%APPDATA%\\koe``. macOS: ``~/Library/Application Support/koe``.
    Otherwise XDG.
    """
    home = Path.home()
    if sys.platform == "win32":
        base = _base_dir("APPDATA", home / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = _base_dir("XDG_CONFIG_HOME", home / ".config")
    return base / APP_NAME


def data_dir() -> Path:
    """Per-user data directory, for anything larger than settings."""
    home = Path.home()
    if sys.platform == "win32":
        base = _base_dir("LOCALAPPDATA", home / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = _base_dir("XDG_DATA_HOME", home / ".local" / "share")
    return base / APP_NAME


def log_dir() -> Path:
    """Where log files go.

    Separate from data on Windows and macOS because a user clearing logs should
    not be able to delete their meetings by accident.
    """
    if sys.platform == "win32":
        return data_dir() / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return _base_dir("XDG_STATE_HOME", Path.home() / ".local" / "state") / APP_NAME / "logs"


def ensure_dirs() -> None:
    """Create the writable directories. Safe to call repeatedly."""
    for path in (config_dir(), data_dir(), log_dir()):
        path.mkdir(parents=True, exist_ok=True)
