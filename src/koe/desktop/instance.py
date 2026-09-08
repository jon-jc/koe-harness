"""Single-instance enforcement.

Two copies of the app would each start a server, each hold a microphone, and
each write the same settings file — and the user would have no way to tell
which window was which. Double-clicking an icon twice is a normal thing for a
person to do, so this has to be handled rather than assumed away.

The lock is a file containing the running process's PID. The subtlety is
**stale locks**: if the app is killed or crashes, the file remains, and a
naive implementation then refuses to ever start again. So a lock whose PID is
no longer alive is taken over rather than respected.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import TracebackType

from koe.desktop.paths import data_dir

logger = logging.getLogger(__name__)

LOCK_FILE = "koe.lock"


class AlreadyRunning(RuntimeError):
    """Another instance holds the lock."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        super().__init__(f"koe is already running (pid {pid})")


#: Beyond this, a value is not a PID any operating system will admit to. The
#: platform calls below reject it by raising rather than returning false, so
#: it has to be filtered before it reaches them.
MAX_PID = 2**31 - 1


def _process_alive(pid: int) -> bool:
    """Whether `pid` names a live process.

    A PID can be recycled, so this can theoretically report a stale lock as
    live. The failure mode is a spurious "already running" message rather than
    two instances corrupting each other's state, which is the right way round.

    The lock file is untrusted input — it can be truncated, hand-edited, or
    left by a different build — so an out-of-range value is treated as "not
    running" rather than passed to a platform call that would raise.
    """
    if pid <= 0 or pid > MAX_PID:
        return False
    # `sys.platform` rather than `os.name`: the type checker narrows on the
    # former and skips this branch entirely off-Windows. With `os.name` it
    # walks in and reports ctypes.windll as missing on Linux and macOS.
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else.
        return True
    except (OverflowError, ValueError):
        # Out of range for this platform's pid_t.
        return False
    return True


class InstanceLock:
    """Holds the single-instance lock for the lifetime of a context."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / LOCK_FILE)
        self._held = False

    def acquire(self) -> None:
        """Take the lock, or raise :class:`AlreadyRunning`.

        Re-acquiring a lock this object already holds is a no-op. Any *other*
        live holder refuses — including one in this same process, because two
        lock objects both believing they own it is a bug worth surfacing rather
        than papering over with a PID comparison.
        """
        if self._held:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        existing = self._read_pid()
        if existing is not None:
            if _process_alive(existing):
                raise AlreadyRunning(existing)
            logger.info("clearing stale lock from pid %d", existing)

        # Not atomic against a simultaneous launch, and deliberately so: the
        # cost of losing that race is two windows, while an OS-level mutex
        # costs a platform-specific dependency for a desktop convenience.
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        self._held = True

    def _read_pid(self) -> int | None:
        """Read the recorded PID, or None if the file is missing or nonsense."""
        try:
            pid = int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return pid if 0 < pid <= MAX_PID else None

    def release(self) -> None:
        """Release the lock, if this process still owns it."""
        if not self._held:
            return
        self._held = False
        # Only remove our own lock: if a stale-lock takeover raced with us,
        # deleting unconditionally would strip the winner's lock.
        if self._read_pid() == os.getpid():
            try:
                self.path.unlink()
            except OSError as exc:
                logger.debug("could not remove lock file: %s", exc)

    def __enter__(self) -> InstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
