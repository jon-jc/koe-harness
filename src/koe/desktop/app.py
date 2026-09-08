"""The koe desktop application.

A native window over the same FastAPI application the server deployment runs,
on loopback. The window is a client of the real API — nothing is stubbed for
desktop — so the two builds cannot drift apart.

Why a webview rather than Electron or a native toolkit:

* The client already exists as TypeScript and is 38 KB. Electron would ship a
  second copy of Chromium (~150 MB) to run it, when Windows 10/11 already has
  WebView2 installed.
* A native toolkit would mean a second implementation of the transcript view,
  which is precisely the thing most likely to diverge from the web one.
* `AudioWorklet`, which the capture path depends on, exists in a webview and
  does not exist in a Python UI toolkit.

The cost is a dependency on the system WebView2 runtime, which is present by
default on Windows 11 and on updated Windows 10. That is checked at startup
and reported as a clear message rather than a stack trace.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from koe.desktop import paths
from koe.desktop.instance import AlreadyRunning, InstanceLock
from koe.desktop.server import EmbeddedServer
from koe.desktop.settings import DesktopSettings, WindowState

logger = logging.getLogger("koe.desktop")

WINDOW_TITLE = "koe 声 — 議事録"
LOG_FILE = "koe.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def configure_desktop_logging(level: str = "INFO") -> None:
    """Log to a rotating file, and to the console when there is one.

    A frozen GUI application has no console, so stderr goes nowhere. Without a
    file, a crash report from a user is "it closed" and nothing else.
    """
    from koe.telemetry.metrics import JSONFormatter

    paths.ensure_dirs()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    file_handler = logging.handlers.RotatingFileHandler(
        paths.log_dir() / LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # `sys.stderr` is None under pythonw / a windowed PyInstaller build.
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(console)

    for noisy in ("uvicorn.access", "multipart", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# error reporting
# --------------------------------------------------------------------------


def show_error(title: str, message: str) -> None:
    """Surface a fatal problem to the user.

    A windowed application that exits silently is indistinguishable from one
    that never launched, so this is a message box where one is available and a
    printed line otherwise.
    """
    logger.error("%s: %s", title, message)
    if sys.platform == "win32":
        try:
            import ctypes

            MB_ICONERROR = 0x10
            ctypes.windll.user32.MessageBoxW(None, message, title, MB_ICONERROR)
            return
        except Exception:  # noqa: BLE001
            # Reporting a failure must not itself fail; fall through to stderr.
            pass
    print(f"{title}: {message}", file=sys.stderr)


def install_crash_handler() -> None:
    """Log and report anything that escapes to the top level."""

    def handle(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        show_error(
            "koe — unexpected error",
            f"{exc_type.__name__}: {exc}\n\nDetails were written to:\n{paths.log_dir()}",
        )

    sys.excepthook = handle

    def handle_thread(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread else "?"
        if args.exc_value is None:
            logger.critical("unhandled exception in thread %s", name)
            return
        logger.critical(
            "unhandled exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = handle_thread


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def prepare_webview_environment() -> None:
    """Configure the embedded browser before it is created.

    The microphone is the reason this exists. WebView2 treats an unhandled
    permission request conservatively, and pywebview registers no handler, so
    without this a user pressing Record gets a silent denial. The flag grants
    media capture for this embedded browser only — it is not a global setting
    and does not affect the user's actual browser.

    This is a genuine trade: the app cannot show its own permission prompt, so
    it grants capture to its own origin and relies on the recording state being
    unmistakable in the UI.
    """
    existing = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
    flags = [
        # Grant getUserMedia without a prompt the app cannot render.
        "--use-fake-ui-for-media-stream",
        # The overscroll animation looks wrong in a fixed-size app window.
        "--disable-features=ElasticOverscroll",
    ]
    merged = " ".join(part for part in [existing, *flags] if part)
    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = merged


def check_webview_available() -> str | None:
    """Return a human-readable problem, or ``None`` if a webview can be used."""
    try:
        import webview  # noqa: F401
    except ImportError:
        return (
            "The desktop UI components are missing from this build.\n\n"
            "Install with: pip install 'koe-harness[desktop]'"
        )
    if sys.platform == "win32":
        import shutil
        from pathlib import Path

        candidates = (
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        )
        edge = bool(shutil.which("msedge")) or any(p.exists() for p in candidates)
        if not edge:
            return (
                "Microsoft Edge WebView2 Runtime was not found.\n\n"
                "koe uses it to render its interface. Install it from:\n"
                "https://developer.microsoft.com/microsoft-edge/webview2/"
            )
    return None


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------


def build_services() -> Any:
    """Assemble the API services with desktop-appropriate defaults."""
    from koe.api.app import Services
    from koe.config import Settings

    settings = Settings(
        environment="local",
        structured_logs=True,
        # One person, one meeting at a time. A high cap would only let a bug
        # exhaust the machine's memory faster.
        max_concurrent_sessions=4,
        cors_origins=["*"],  # loopback only; the window is the sole client
    )
    return Services.default(settings)


class DesktopApp:
    """Owns the window, the embedded server, and the lock."""

    def __init__(self) -> None:
        self.settings = DesktopSettings.load()
        self.server: EmbeddedServer | None = None
        self.window: Any = None

    def run(self) -> int:
        from koe.api.app import create_app

        services = build_services()
        self.server = EmbeddedServer(create_app(services))
        self.server.start()
        logger.info("embedded server ready", extra={"url": self.server.url})

        if os.environ.get("KOE_DESKTOP_HEADLESS"):
            return self._run_headless()

        import webview

        state = self.settings.window.sanitized()
        self.window = webview.create_window(
            WINDOW_TITLE,
            url=self.server.url,
            width=state.width,
            height=state.height,
            x=state.x,
            y=state.y,
            min_size=(900, 600),
            confirm_close=False,
        )

        self.window.events.loaded += self._on_loaded
        self.window.events.resized += self._on_resized
        self.window.events.moved += self._on_moved
        self.window.events.closing += self._on_closing

        # `private_mode=False` with an explicit storage path gives the page
        # persistent localStorage, which is what makes the theme choice stick
        # between launches.
        webview.start(
            private_mode=False,
            storage_path=str(paths.data_dir() / "webview"),
            debug=bool(os.environ.get("KOE_DESKTOP_DEBUG")),
        )
        return 0

    def _run_headless(self) -> int:
        """Serve without opening a window.

        The packaging step uses this to prove a *frozen* build actually runs.
        That matters because PyInstaller fails at import time far more often
        than at build time: a missing hidden import produces a successful build
        and an executable that closes instantly, and the only way to catch it
        is to start the thing. It also makes the artifact testable in CI, where
        there is no display.
        """
        assert self.server is not None
        port_file = os.environ.get("KOE_DESKTOP_PORT_FILE")
        if port_file:
            path = Path(port_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(self.server.port), encoding="utf-8")

        logger.info("running headless", extra={"url": self.server.url})
        try:
            while self.server.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
        return 0

    # -- window events -------------------------------------------------------

    def _on_loaded(self) -> None:
        """Record what the embedded browser can actually do.

        Support for a desktop app is somebody describing a symptom over email,
        so the log has to answer the first question — "could it even reach the
        microphone?" — without a round trip. This inspects capability only; it
        deliberately does not call `getUserMedia`, which would switch the
        microphone on without the user asking.
        """
        probe = """
        (function () {
          return JSON.stringify({
            secureContext: window.isSecureContext === true,
            mediaDevices: !!(navigator.mediaDevices &&
                             navigator.mediaDevices.getUserMedia),
            audioWorklet: typeof AudioWorklet !== 'undefined',
            webSocket: typeof WebSocket !== 'undefined',
            userAgent: navigator.userAgent
          });
        })()
        """
        try:
            import json

            raw = self.window.evaluate_js(probe)
            capabilities = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break startup
            logger.warning("capability probe failed: %s", exc)
            return

        logger.info("webview capabilities", extra=dict(capabilities))
        if not capabilities.get("mediaDevices"):
            # Recording will not work; the demo still will. Worth saying once,
            # loudly, rather than leaving the user to discover it by pressing
            # a button that silently does nothing.
            logger.error(
                "microphone capture is unavailable in this webview; demo playback will still work"
            )

    def _on_resized(self, width: int, height: int) -> None:
        self.settings.window = WindowState(
            width=int(width),
            height=int(height),
            x=self.settings.window.x,
            y=self.settings.window.y,
        )

    def _on_moved(self, x: int, y: int) -> None:
        self.settings.window = WindowState(
            width=self.settings.window.width,
            height=self.settings.window.height,
            x=int(x),
            y=int(y),
        )

    def _on_closing(self) -> bool:
        logger.info("window closing")
        self.settings.save()
        if self.server is not None:
            self.server.stop()
        return True


def main(argv: list[str] | None = None) -> int:
    """Entry point for the desktop application."""
    configure_desktop_logging(os.environ.get("KOE_LOG_LEVEL", "INFO"))
    install_crash_handler()

    from koe import __version__

    logger.info(
        "koe desktop starting",
        extra={"version": __version__, "frozen": paths.is_frozen()},
    )

    problem = check_webview_available()
    if problem:
        show_error("koe — cannot start", problem)
        return 1

    prepare_webview_environment()

    lock = InstanceLock()
    try:
        lock.acquire()
    except AlreadyRunning as exc:
        show_error(
            "koe is already running",
            f"Another copy of koe is open (process {exc.pid}).\n\nSwitch to that window instead.",
        )
        return 1

    try:
        return DesktopApp().run()
    except Exception as exc:
        logger.exception("desktop application failed")
        show_error(
            "koe — unexpected error",
            f"{type(exc).__name__}: {exc}\n\nDetails were written to:\n{paths.log_dir()}",
        )
        return 1
    finally:
        lock.release()
        logger.info("koe desktop stopped")


if __name__ == "__main__":
    sys.exit(main())
