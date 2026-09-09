"""A hidden console, so ConPTY has something to attach to.

The desktop application is built with ``console=False`` — a console window
flashing up behind a GUI app looks broken, and PyInstaller's GUI subsystem is
the right default. But ConPTY is a *console* API: ``CreatePseudoConsole``
needs the calling process to have one, and in a GUI-subsystem process there is
none.

The symptom is precise and misleading. The pty spawns, bash starts and even
sets the window title, and then the child dies with ``0xC000013A`` —
``STATUS_CONTROL_C_EXIT``. It reads exactly like the user pressed Ctrl+C, so
the natural place to look is signal handling, which is the wrong place
entirely. The same code works perfectly under ``python.exe``, because a
console session has a console.

So the app allocates one and immediately hides its window. This is what
GUI applications that host terminals do; there is no console-less ConPTY.

Nothing else in koe needs this, which is why it is here rather than in the
terminal package: the terminal is a library that works fine in any process
that already has a console, and the *frozen GUI application* is the one
deployment that does not.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

#: ShowWindow(SW_HIDE).
_SW_HIDE = 0


def ensure_console() -> bool:
    """Attach a hidden console if this process has none.

    Returns True when a console is available afterwards, either because one
    already existed or because this call made one. False means the terminal's
    pty backend will not work here — which the capabilities endpoint reports
    rather than discovering at the first keystroke.

    Safe to call more than once, and a no-op off Windows.
    """
    if sys.platform != "win32":
        return True

    import ctypes

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    existing = kernel32.GetConsoleWindow()
    if existing:
        # A console session, or a second call. Leave the window alone: it may
        # be the terminal the user launched us from.
        return True

    if not kernel32.AllocConsole():
        logger.warning("could not allocate a console; the pty terminal will be unavailable")
        return False

    window = kernel32.GetConsoleWindow()
    if window:
        # Hidden rather than closed: closing the console window sends
        # CTRL_CLOSE_EVENT to the whole process group, which would take the
        # app down with it.
        user32.ShowWindow(window, _SW_HIDE)

    logger.info("allocated a hidden console for ConPTY")
    return True
