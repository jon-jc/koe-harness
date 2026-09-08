"""The embedded API server.

The desktop app runs the same FastAPI application the server deployment does,
on loopback, inside its own process. Nothing is stubbed or reimplemented for
desktop — the window is a client of the same API, which means the desktop build
cannot quietly drift from the deployed one.

Three details that are the whole reason this is a module and not four lines:

**The socket is bound before the server starts.** Asking the OS for port 0 and
reading back the assignment means the window knows the URL with certainty. The
alternative — start on a guessed port, poll until something answers — has a
race on startup and a conflict whenever the guess is taken, and both failures
look like "the app didn't open".

**Startup is awaited, not slept on.** The window must not load a URL before
something is listening, and a fixed sleep is either too short on a cold start
or wasted time on a warm one.

**Shutdown is graceful and bounded.** uvicorn is asked to stop and given time
to drain in-flight sessions; past the deadline the process exits anyway,
because a GUI app that will not close is worse than one that drops a session.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
from typing import Any

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 10.0


def reserve_loopback_port() -> tuple[socket.socket, int]:
    """Bind a loopback socket on an OS-assigned port.

    Returns the bound socket and its port. The socket is handed to uvicorn
    rather than closed, so nothing else can take the port in between.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Deliberately no SO_REUSEADDR: on Windows that permits two processes to
    # bind the same port, which is exactly the collision being avoided here.
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = int(sock.getsockname()[1])
    return sock, port


class EmbeddedServer:
    """Runs the koe API on loopback in a background thread."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._socket, self.port = reserve_loopback_port()
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the server and block until it is accepting connections."""
        import uvicorn

        config = uvicorn.Config(
            app=self._app,
            log_config=None,  # the app installs its own structured logging
            access_log=False,
            timeout_graceful_shutdown=int(SHUTDOWN_TIMEOUT),
        )
        self._server = uvicorn.Server(config)
        # uvicorn installs signal handlers by default, which is wrong in a GUI
        # process: it is not the owner of the process's signal disposition, and
        # on a non-main thread the install fails outright.
        self._server.install_signal_handlers = lambda: None

        self._thread = threading.Thread(target=self._run, name="koe.server", daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=STARTUP_TIMEOUT):
            raise RuntimeError(f"server did not start within {STARTUP_TIMEOUT:.0f}s")
        if self._failure is not None:
            raise RuntimeError(f"server failed to start: {self._failure}") from self._failure

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # reported back through start()
            self._failure = exc
            logger.exception("embedded server crashed")
        finally:
            # Release the readiness gate even on failure, or start() waits the
            # full timeout to report an error that already happened.
            self._ready.set()
            with contextlib.suppress(Exception):
                loop.close()

    async def _serve(self) -> None:
        assert self._server is not None
        serve_task = asyncio.create_task(self._server.serve(sockets=[self._socket]))

        # uvicorn sets `started` once the application lifespan has completed,
        # which is the point at which the app is genuinely ready — not merely
        # when the socket is listening.
        deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
        while not getattr(self._server, "started", False):
            if serve_task.done():
                break
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.02)

        self._ready.set()
        await serve_task

    def stop(self) -> None:
        """Ask the server to shut down and wait, bounded."""
        if self._server is None or self._loop is None:
            return
        logger.info("stopping embedded server")
        self._server.should_exit = True

        thread = self._thread
        if thread is not None:
            thread.join(timeout=SHUTDOWN_TIMEOUT)
            if thread.is_alive():
                # The process is exiting regardless. A GUI that refuses to
                # close is a worse outcome than a dropped session.
                logger.warning("server did not stop within %.0fs", SHUTDOWN_TIMEOUT)
        with contextlib.suppress(Exception):
            self._socket.close()

    def __enter__(self) -> EmbeddedServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
