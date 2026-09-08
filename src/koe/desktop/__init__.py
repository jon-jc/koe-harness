"""The koe desktop application.

A native window over the same FastAPI application the server deployment runs.
See :mod:`koe.desktop.app` for why a webview rather than Electron.
"""

from koe.desktop.app import DesktopApp, main
from koe.desktop.instance import AlreadyRunning, InstanceLock
from koe.desktop.server import EmbeddedServer
from koe.desktop.settings import DesktopSettings, WindowState

__all__ = [
    "AlreadyRunning",
    "DesktopApp",
    "DesktopSettings",
    "EmbeddedServer",
    "InstanceLock",
    "WindowState",
    "main",
]
