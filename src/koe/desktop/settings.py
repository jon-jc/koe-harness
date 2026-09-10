"""Persisted desktop preferences.

Deliberately separate from :mod:`koe.config`, which is the *service*
configuration and comes from the environment. These are *user* preferences that
the app writes back — window geometry, theme, language — and the two have
different lifecycles: an operator sets the former once per deployment, a person
changes the latter by resizing a window.

Every read is defensive. A settings file can be truncated by a power loss,
hand-edited into invalid JSON, or written by a newer version with fields this
one does not know. None of those should stop the app from opening; the worst
acceptable outcome is that a window comes back at its default size.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from koe.desktop.paths import config_dir

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"

#: Below this a window is unusable; above the largest plausible display it is
#: almost certainly a stale value from a monitor that is no longer attached.
MIN_WINDOW = (720, 480)
MAX_WINDOW = (10000, 10000)


@dataclass(slots=True)
class WindowState:
    width: int = 1280
    height: int = 860
    x: int | None = None
    y: int | None = None
    maximized: bool = False

    def sanitized(self) -> WindowState:
        """Clamp to something that can actually be displayed.

        Restoring a window to a position on a monitor that has since been
        unplugged puts it off-screen, where it is running but invisible and the
        user concludes the app failed to start. Position is dropped rather than
        guessed, which re-centres it.
        """
        width = min(max(self.width, MIN_WINDOW[0]), MAX_WINDOW[0])
        height = min(max(self.height, MIN_WINDOW[1]), MAX_WINDOW[1])
        x, y = self.x, self.y
        if x is not None and (x < -width or x > MAX_WINDOW[0]):
            x = None
        if y is not None and (y < -height or y > MAX_WINDOW[1]):
            y = None
        return WindowState(width=width, height=height, x=x, y=y, maximized=self.maximized)


@dataclass(slots=True)
class DesktopSettings:
    """What the app remembers between launches."""

    window: WindowState = field(default_factory=WindowState)
    #: Interface language. Recognition language is a separate, in-page control.
    ui_language: str = "ja"
    theme: str = "system"
    #: Opt-in to keeping meetings on disk. Off by default: a tool that starts
    #: silently recording meetings to disk is not a decision to make on a
    #: user's behalf.
    save_meetings: bool = False
    #: Download new releases in the background and install them as the app
    #: closes. On by default — a desktop tool weeks behind its own fixes is the
    #: usual way one rots — and one switch away in Settings → About.
    auto_update: bool = True

    @classmethod
    def path(cls) -> Path:
        return config_dir() / SETTINGS_FILE

    @classmethod
    def load(cls) -> DesktopSettings:
        """Read settings, falling back to defaults on any problem."""
        path = cls.path()
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("settings unreadable (%s); using defaults", exc)
            return cls()
        if not isinstance(raw, dict):
            logger.warning("settings is not an object; using defaults")
            return cls()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DesktopSettings:
        """Build from a mapping, ignoring unknown keys and bad types.

        Unknown keys are dropped rather than rejected so that a file written by
        a newer version still opens in an older one.
        """
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in raw.items() if k in known}

        window = WindowState()
        raw_window = data.pop("window", None)
        if isinstance(raw_window, dict):
            window_fields = {f.name for f in fields(WindowState)}
            try:
                window = WindowState(**{k: v for k, v in raw_window.items() if k in window_fields})
            except TypeError as exc:
                logger.warning("window state invalid (%s); using defaults", exc)

        settings = cls(window=window.sanitized())
        for key, value in data.items():
            if isinstance(value, type(getattr(settings, key))):
                setattr(settings, key, value)
        return settings

    def save(self) -> None:
        """Write settings atomically.

        Via a temporary file and a replace, because a crash partway through a
        direct write leaves truncated JSON — and the next launch would then
        lose every preference rather than the one being changed.
        """
        path = self.path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            # Never let a failed preference write take the app down; the user
            # is mid-session and losing a window size is not worth a crash.
            logger.warning("could not save settings: %s", exc)
