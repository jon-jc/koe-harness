"""Two ways to talk to a shell, behind one interface.

The terminal originally spoke to an ``asyncio.subprocess.Process`` directly.
That was fine while pipes were the only transport and wrong the moment a pty
arrived, because a pty is not a process with three streams — it is one
bidirectional file descriptor that also carries a window size. Rather than
branch on which one we have in every method, both become a
:class:`Channel`, and the session never learns the difference.

## Why both exist

They are not a fallback pair. They serve different consumers and each is
better than the other at its job.

**Pipes are for the model.** No escape sequences, no cursor addressing, a
prompt sentinel we control, and output that costs what it says it costs. A pty
would hand a model a screenful of ANSI to pay for and reason about.

**A pty is for the person.** `vim`, `top`, `less` and every program that draws
rather than prints needs a terminal on the other end: they check `isatty`, ask
for the window size, and address the cursor. Pipes make them degrade or refuse.
That output is only useful to something that can render it, which is why the
pty path exists alongside an emulator in the client and not instead of the
pipe path.

## What a pty costs

It is a real dependency and a real platform split. Windows needs ConPTY
through `pywinpty`; POSIX has `pty` in the standard library. Neither is
asyncio-native — the Windows read is a blocking call in a thread, and the
POSIX one is a raw file descriptor the loop can watch. Both are absent from a
minimal install, so `available()` answers honestly and the caller falls back to
pipes rather than failing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Read size. Large enough that a full-screen redraw arrives in one or two
#: reads, small enough not to sit on a partial frame waiting to fill a buffer.
CHUNK = 8192


@runtime_checkable
class Channel(Protocol):
    """A bidirectional byte stream to a running process."""

    @property
    def pid(self) -> int: ...

    @property
    def alive(self) -> bool: ...

    @property
    def exit_code(self) -> int | None: ...

    async def read(self) -> bytes:
        """The next chunk, or ``b""`` at end of stream."""
        ...

    async def write(self, data: bytes) -> None: ...

    def resize(self, cols: int, rows: int) -> None:
        """Tell the process its window changed. A no-op where there is none."""
        ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------
# pipes
# --------------------------------------------------------------------------


class PipeChannel:
    """An ``asyncio.subprocess.Process`` with stdout and stderr merged."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def alive(self) -> bool:
        return self._process.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self._process.returncode

    async def read(self) -> bytes:
        stream = self._process.stdout
        if stream is None:
            return b""
        return await stream.read(CHUNK)

    async def write(self, data: bytes) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise BrokenPipeError("the process has no stdin")
        stdin.write(data)
        await stdin.drain()

    def resize(self, cols: int, rows: int) -> None:
        """Nothing to resize. A pipe has no window, and pretending otherwise
        would let a caller believe a program was told about a change it was
        not."""

    def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.terminate()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()

    async def wait(self) -> int:
        return await self._process.wait()

    async def aclose(self) -> None:
        # Closing stdin first: a shell reading a closed stdin exits on its own,
        # which is cleaner than a signal — and leaving the pipe open means
        # asyncio finalizes the transport during garbage collection, after the
        # loop has gone, which surfaces as "Event loop is closed" from a
        # destructor nobody can catch.
        if self._process.stdin is not None:
            with contextlib.suppress(Exception):
                self._process.stdin.close()


# --------------------------------------------------------------------------
# pty
# --------------------------------------------------------------------------


def pty_available() -> bool:
    """Whether a pty can be opened on this machine."""
    if sys.platform == "win32":
        try:
            import winpty  # noqa: F401
        except ImportError:
            return False
        return True
    return hasattr(os, "openpty")


class WindowsPtyChannel:
    """ConPTY, through pywinpty.

    `pywinpty` is synchronous and its read blocks, so every call goes to a
    worker thread. One thread per terminal is the right trade here: a terminal
    is a human-scale object — a handful at once, alive for minutes — and the
    alternative is polling, which trades latency for the same thread.
    """

    def __init__(self, process: Any) -> None:
        self._process = process
        self._exit: int | None = None
        self._closed = False

    @property
    def pid(self) -> int:
        return int(getattr(self._process, "pid", 0) or 0)

    @property
    def alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self._process.isalive())
        except Exception:  # noqa: BLE001 - a dead handle means dead
            return False

    @property
    def exit_code(self) -> int | None:
        if self.alive:
            return None
        if self._exit is None:
            with contextlib.suppress(Exception):
                self._exit = self._process.exitstatus
        return self._exit

    async def read(self) -> bytes:
        def _read() -> bytes:
            try:
                # pywinpty decodes for us; the session wants bytes so the two
                # channels are interchangeable.
                text: str = self._process.read(CHUNK)
                return text.encode("utf-8", "replace")
            except EOFError:
                return b""
            except Exception:  # noqa: BLE001 - a closed pty reads as EOF
                return b""

        if self._closed:
            return b""
        return await asyncio.to_thread(_read)

    async def write(self, data: bytes) -> None:
        text = data.decode("utf-8", "replace")
        await asyncio.to_thread(self._process.write, text)

    def resize(self, cols: int, rows: int) -> None:
        with contextlib.suppress(Exception):
            # pywinpty takes (rows, cols); getting that backwards makes vim
            # wrap at the wrong column, which looks like a rendering bug
            # anywhere except here.
            self._process.setwinsize(rows, cols)

    def terminate(self) -> None:
        with contextlib.suppress(Exception):
            self._process.terminate()

    def kill(self) -> None:
        with contextlib.suppress(Exception):
            self._process.terminate(force=True)

    async def wait(self) -> int:
        def _wait() -> int:
            with contextlib.suppress(Exception):
                return int(self._process.wait())
            return -1

        self._exit = await asyncio.to_thread(_wait)
        return self._exit

    async def aclose(self) -> None:
        self._closed = True
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self._process.close)


class PosixPtyChannel:
    """A pty from the standard library, with the child as a subprocess.

    The master descriptor is read in a thread rather than through
    ``loop.connect_read_pipe``: a pty master reports EOF as an ``EIO`` errno
    rather than an empty read, and the transport machinery treats that as a
    fatal error on a socket it then tries to close twice. Reading it directly
    keeps that quirk in one place, where it is explained.
    """

    def __init__(self, process: asyncio.subprocess.Process, master: int) -> None:
        self._process = process
        self._master = master
        self._closed = False

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def alive(self) -> bool:
        return self._process.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self._process.returncode

    async def read(self) -> bytes:
        def _read() -> bytes:
            try:
                return os.read(self._master, CHUNK)
            except OSError:
                # EIO on a pty master means the child closed the slave: end of
                # stream, not a failure.
                return b""

        if self._closed:
            return b""
        return await asyncio.to_thread(_read)

    async def write(self, data: bytes) -> None:
        await asyncio.to_thread(os.write, self._master, data)

    def resize(self, cols: int, rows: int) -> None:
        # The platform check is what lets this typecheck on both: fcntl and
        # termios do not exist on Windows, and mypy narrows on `sys.platform`,
        # so the rest of the body is unreachable there rather than an error.
        # A blanket `type: ignore` would be flagged as unused on Linux, which
        # is the same problem wearing a different hat.
        if sys.platform == "win32":
            return
        try:
            import fcntl
            import struct
            import termios

            packed = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self._master, termios.TIOCSWINSZ, packed)
        except OSError:
            # A resize that fails is cosmetic: the program keeps running at
            # the size it had.
            logger.debug("could not resize pty")

    def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.terminate()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()

    async def wait(self) -> int:
        return await self._process.wait()

    async def aclose(self) -> None:
        self._closed = True
        with contextlib.suppress(OSError):
            os.close(self._master)


async def open_pty(
    command: list[str], cwd: str, env: dict[str, str], size: tuple[int, int]
) -> Channel:
    """Start `command` attached to a pty. Raises if this platform has none."""
    cols, rows = size
    if sys.platform == "win32":
        from winpty import PtyProcess

        process = await asyncio.to_thread(
            PtyProcess.spawn, command, cwd=cwd, env=env, dimensions=(rows, cols)
        )
        return WindowsPtyChannel(process)

    master, slave = os.openpty()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        # The parent's copy of the slave has to go, or the master never sees
        # EOF when the child exits: the descriptor stays open and every read
        # blocks forever on a shell that is already gone.
        os.close(slave)

    channel = PosixPtyChannel(process, master)
    channel.resize(cols, rows)
    return channel
