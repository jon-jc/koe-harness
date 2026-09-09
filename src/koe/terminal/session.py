"""Persistent terminal sessions.

A terminal is the one capability where "run a command and return the output" is
not enough. The work people actually do in one is *stateful*: `cd` into a
directory and stay there, activate a virtualenv, step a debugger, sit in a REPL.
One-shot execution loses all of it between calls, so the unit here is a session
that outlives the command.

Four rules, three of them taken from how dsh scopes the same capability.

**A session is owned, and the id is not a capability.** Every operation naming
a session is refused unless the caller is its owner. Learning an id from a log
is not authorization; this is what keeps one agent out of another's debugger.

**One send at a time.** A second send while the first is still settling is
refused rather than interleaved, because two writers on one stdin produce a
line neither of them meant to type.

**A send settles with a reason, and the good reason is not a guess.** The
shell is started with a prompt we chose, so the appearance of that sentinel is
proof the command returned and the shell is waiting for input — ``prompt``.
Everything else is weaker: ``inferred_idle`` means output merely went quiet,
which a command that pauses mid-work also does; ``timeout`` means the budget
ran out; ``session_exit`` means the shell is gone. A caller that cannot tell
"finished" from "still working" will either truncate output or hang, and only
one of those is recoverable — so the distinction is in the result rather than
inferred by whoever reads it.

**Output is a bounded ring.** A command that prints forever must not become a
memory leak, so the buffer keeps the most recent bytes and says how much it
dropped. The end of a runaway command is the part anyone wants.

## Cleaning

Output is stripped of ANSI escapes, the echoed command, and the trailing
prompt. A model pays by the token for a colour code, and a plain-text UI
renders it as garbage; the shell's own prompt is noise in both. Dropping colour
is a real loss and a deliberate one — rendering it properly needs a terminal
emulator in the client, which is a much larger dependency than the benefit.

The shell is started with ``--norc --noprofile`` so its prompt is the one we
set rather than whatever the user's rc file overrides it with. That also costs
the user's aliases, which for a harness is the right trade: a terminal that
behaves the same on every machine is worth more here than one that feels like
home on this one.

## What this is not

There is no pty. Output is read from pipes, which covers everything
non-interactive — builds, tests, git, REPL turns — and does not cover
full-screen programs that drive a terminal directly (``vim``, ``top``,
``htop``). Those see a non-tty stdout and either degrade or refuse. Adding
ConPTY on Windows and ``pty`` elsewhere is a backend change, which is why
backends are a seam rather than a branch in this file.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Bytes of output retained per session.
BUFFER_BYTES = 256_000

#: How often the waiter wakes to re-check for the prompt.
IDLE_MS = 120.0

#: How long silence must last before a send settles *without* a prompt.
#:
#: Deliberately long. When the shell has a sentinel prompt, silence is not
#: evidence of anything -- `sleep 3` is silent and unfinished -- so the only
#: reason to settle on it at all is a sub-prompt the sentinel never reaches:
#: a REPL, or a command waiting on input. Those are quiet for as long as
#: nobody types, and a normal command that merely pauses is not quiet this
#: long. A short grace here reports "done" for every slow command.
IDLE_GRACE_MS = 2500.0

#: Default ceiling on one send.
SEND_TIMEOUT_S = 30.0

#: A session nobody has touched for this long is reaped.
IDLE_SESSION_S = 3600.0

#: The prompt the shell is told to print. Unlikely in real output, and the
#: reason a send can settle on proof rather than on silence.
PROMPT_SENTINEL = "@@koe@@"


#: CSI sequences (colour, cursor moves) and OSC sequences (window title).
#: Two patterns rather than one because OSC runs to a BEL or an ST rather than
#: to a letter, and folding them into one expression makes both unreadable.
_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(text: str) -> str:
    """Remove escape sequences and lone carriage returns.

    A model pays by the token for a colour code and a plain-text view renders
    it as garbage, so neither consumer wants them. The `\r` removal matters
    separately: progress output that rewrites one line with carriage returns
    reads as a single unbroken line once the escapes are gone.
    """
    text = _OSC.sub("", _CSI.sub("", text))
    return text.replace("\r\n", "\n").replace("\r", "")


def clean_output(text: str, *, command: str = "") -> str:
    """Strip the sentinel prompt and the shell's echo of the command.

    Both are things the caller already has: they typed the command, and the
    prompt is ours. Returning them costs tokens and reads as noise.
    """
    text = strip_ansi(text)
    lines = text.split("\n")

    if command:
        # Interactive bash on a pipe echoes what it read. It is always the
        # first line, so an exact match is enough and a search is not.
        wanted = command.strip()
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].strip() == wanted:
            lines.pop(0)

    lines = [line for line in lines if line.strip() != PROMPT_SENTINEL]
    return "\n".join(lines).strip("\n")


class TerminalError(StrEnum):
    """Stable codes, routable by both the model and the browser."""

    NO_BACKEND = "no_backend"
    NO_SESSION = "no_session"
    FOREIGN_SESSION = "foreign_session"
    SEND_ACTIVE = "send_active"
    SESSION_EXITED = "session_exited"
    SPAWN_FAILED = "spawn_failed"
    TOO_MANY = "too_many"


class WaitReason(StrEnum):
    """Why a send stopped waiting. The caller's cue for what to do next."""

    #: The shell printed its prompt: the command returned. Proof, not a guess.
    PROMPT = "prompt"
    #: Output went quiet. A fallback, and a guess -- a command that pauses
    #: longer than the idle window settles early.
    IDLE = "inferred_idle"
    #: The wait budget ran out; the command is still running.
    TIMEOUT = "timeout"
    #: The shell itself exited.
    SESSION_EXIT = "session_exit"


class TerminalFailure(Exception):
    """A refused or impossible terminal operation."""

    def __init__(self, code: TerminalError, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class SendOutcome:
    """What one send produced."""

    output: str
    reason: WaitReason
    duration_ms: float
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "reason": self.reason.value,
            "duration_ms": round(self.duration_ms, 1),
            "exit_code": self.exit_code,
        }


class Backend(Protocol):
    """Spawns the process behind a session.

    A protocol rather than a base class so a ConPTY backend, a container
    backend, or a remote one can be dropped in without touching the service.
    """

    #: The stable name callers open sessions by.
    type: str

    async def spawn(self, cwd: Path) -> asyncio.subprocess.Process: ...


class ShellBackend:
    """The local shell, over pipes."""

    type = "shell"

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or self._default_command()

    @staticmethod
    def _default_command() -> list[str]:
        """The shell to run here.

        On Windows, prefer git-bash when it is present: the rest of koe's
        tooling assumes POSIX utilities, and a user who has git installed has
        them. `cmd.exe` is the fallback that always exists.
        """
        if sys.platform == "win32":
            for candidate in ("bash.exe", "bash"):
                found = shutil.which(candidate)
                if found:
                    return [found, "--norc", "--noprofile", "-i"]
            return [os.environ.get("COMSPEC", "cmd.exe")]
        return [os.environ.get("SHELL", "/bin/bash"), "--norc", "--noprofile", "-i"]

    async def spawn(self, cwd: Path) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        # A prompt we can recognise, and no colour: escape sequences in a
        # buffer that is going to a model are noise it pays for by the token.
        # PS1 only survives because the shell is started with --norc.
        env["PS1"] = f"{PROMPT_SENTINEL}\n"
        env["PS2"] = ""
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        return await asyncio.create_subprocess_exec(
            *self._command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )


@dataclass
class Session:
    """One live terminal."""

    id: str
    owner: str
    type: str
    name: str
    cwd: str
    process: asyncio.subprocess.Process
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    #: Most recent output. A deque of chunks with a byte budget, so a runaway
    #: command costs a bounded amount and keeps its most recent output.
    _buffer: deque[str] = field(default_factory=deque, repr=False)
    _buffered: int = 0
    dropped: int = 0
    #: Set while a send is in flight, so a second one can be refused.
    _sending: bool = False
    #: Signalled by the reader whenever new output lands.
    _activity: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _reader: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def alive(self) -> bool:
        return self.process.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self.process.returncode

    def append(self, text: str) -> None:
        self._buffer.append(text)
        self._buffered += len(text)
        while self._buffered > BUFFER_BYTES and len(self._buffer) > 1:
            gone = self._buffer.popleft()
            self._buffered -= len(gone)
            self.dropped += len(gone)
        self._activity.set()

    def drain(self) -> str:
        """Take everything buffered and clear it."""
        text = "".join(self._buffer)
        self._buffer.clear()
        self._buffered = 0
        return text

    def peek(self) -> str:
        return "".join(self._buffer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "type": self.type,
            "name": self.name,
            "cwd": self.cwd,
            "pid": self.process.pid,
            "alive": self.alive,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "dropped_bytes": self.dropped,
        }


class TerminalService:
    """Mints sessions, routes to backends, and fences every operation by owner."""

    def __init__(self, *, cwd: Path | None = None, max_sessions: int = 8) -> None:
        self._sessions: dict[str, Session] = {}
        self._backends: dict[str, Backend] = {}
        self._cwd = Path(cwd or Path.cwd())
        self._max = max_sessions

    # -- backends ----------------------------------------------------------

    def register_backend(self, backend: Backend) -> None:
        self._backends[backend.type] = backend

    # -- lifecycle ---------------------------------------------------------

    async def open(self, *, owner: str = "default", type: str = "shell", name: str = "") -> Session:
        backend = self._backends.get(type)
        if backend is None:
            known = ", ".join(sorted(self._backends)) or "none"
            raise TerminalFailure(
                TerminalError.NO_BACKEND, f"no backend of type {type!r} (have: {known})"
            )

        live = [s for s in self._sessions.values() if s.alive]
        if len(live) >= self._max:
            # Refuse rather than evict: the caller knows which of their
            # sessions is finished and we do not.
            raise TerminalFailure(
                TerminalError.TOO_MANY, f"already running {len(live)} sessions (max {self._max})"
            )

        try:
            process = await backend.spawn(self._cwd)
        except (OSError, ValueError) as exc:
            raise TerminalFailure(
                TerminalError.SPAWN_FAILED, f"could not start a {type} session: {exc}"
            ) from exc

        session = Session(
            id=f"t_{uuid.uuid4().hex[:12]}",
            owner=owner,
            type=type,
            name=name or f"session {len(self._sessions) + 1}",
            cwd=str(self._cwd),
            process=process,
        )
        session._reader = asyncio.create_task(self._pump(session))
        self._sessions[session.id] = session

        # Wait for the first prompt, so the caller gets a settled terminal
        # rather than a banner arriving in the middle of their first command.
        await self._quiet(session, idle_ms=IDLE_MS, budget_s=5.0)
        session.drain()
        logger.info("terminal opened", extra={"session": session.id, "owner": owner})
        return session

    async def _pump(self, session: Session) -> None:
        """Continuously move process output into the session buffer.

        Runs for the life of the session rather than only during a send: a
        command that keeps printing after a send settles still has its output
        captured, which is the difference between a background build you can
        read later and one whose output is silently discarded.
        """
        stream = session.process.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                session.append(chunk.decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a broken pipe ends the session, not the server
            logger.debug("terminal reader stopped", extra={"session": session.id})
        finally:
            # Wake anything waiting, so a send does not sit out its full
            # budget after the shell has already gone.
            session._activity.set()

    async def close(self, session_id: str, *, owner: str = "default") -> Session:
        session = self._require(session_id, owner)
        await self._terminate(session)
        return session

    async def _terminate(self, session: Session) -> None:
        if session._reader is not None:
            session._reader.cancel()
            with_suppressed = asyncio.gather(session._reader, return_exceptions=True)
            await with_suppressed
        if session.alive:
            session.process.terminate()
            try:
                await asyncio.wait_for(session.process.wait(), timeout=5.0)
            except TimeoutError:
                # A shell ignoring SIGTERM is usually one with a foreground
                # child that ignored it too. Killing the shell is what we can
                # reach from here.
                session.process.kill()
                await session.process.wait()

    async def dispose(self) -> None:
        """Close every session. Called when the service's scope unwinds."""
        for session in list(self._sessions.values()):
            try:
                await self._terminate(session)
            except Exception:
                logger.exception("failed to close terminal %s", session.id)
        self._sessions.clear()

    # -- interaction -------------------------------------------------------

    async def send(
        self,
        session_id: str,
        text: str,
        *,
        owner: str = "default",
        submit: bool = True,
        timeout_s: float = SEND_TIMEOUT_S,
    ) -> SendOutcome:
        """Write to the session and wait for it to go quiet."""
        session = self._require(session_id, owner)
        if not session.alive:
            raise TerminalFailure(
                TerminalError.SESSION_EXITED,
                f"session exited with code {session.exit_code}",
            )
        if session._sending:
            raise TerminalFailure(
                TerminalError.SEND_ACTIVE, "another send is still settling on this session"
            )
        if session.process.stdin is None:
            raise TerminalFailure(TerminalError.SESSION_EXITED, "session has no stdin")

        session._sending = True
        session.last_used = time.time()
        started = time.perf_counter()
        try:
            # Drain first: output from before this send belongs to whatever
            # produced it, not to the command about to run.
            session.drain()
            payload = text + ("\n" if submit and not text.endswith("\n") else "")
            session.process.stdin.write(payload.encode("utf-8"))
            await session.process.stdin.drain()
            reason = await self._quiet(session, idle_ms=IDLE_MS, budget_s=timeout_s)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise TerminalFailure(
                TerminalError.SESSION_EXITED, f"session closed while writing: {exc}"
            ) from exc
        finally:
            session._sending = False

        return SendOutcome(
            output=clean_output(session.drain(), command=text),
            reason=reason,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            exit_code=session.exit_code,
        )

    async def _quiet(
        self, session: Session, *, idle_ms: float, budget_s: float, expect_prompt: bool = True
    ) -> WaitReason:
        """Wait for the prompt, for silence, for the budget, or for the exit.

        The prompt is checked first and on every wake, because it is the only
        one of the four that is evidence rather than inference: the shell
        prints it when it is ready for the next command, so seeing it means the
        previous one returned.
        """
        deadline = time.monotonic() + budget_s
        quiet_since = time.monotonic()
        grace_s = IDLE_GRACE_MS / 1000.0

        while True:
            if expect_prompt and PROMPT_SENTINEL in session.peek():
                return WaitReason.PROMPT
            if not session.alive:
                return WaitReason.SESSION_EXIT

            now = time.monotonic()
            if now >= deadline:
                return WaitReason.TIMEOUT
            # Settling on silence alone is the weak path, and it is only
            # reached after a long quiet: a REPL or a command waiting on input
            # stays silent indefinitely, where a slow command does not.
            if now - quiet_since >= grace_s:
                return WaitReason.IDLE

            session._activity.clear()
            try:
                await asyncio.wait_for(
                    session._activity.wait(),
                    timeout=min(idle_ms / 1000.0, deadline - now),
                )
            except TimeoutError:
                continue
            quiet_since = time.monotonic()

    def read(self, session_id: str, *, owner: str = "default") -> str:
        """Whatever has accumulated since the last read, without waiting.

        Useful after a `timeout`: the command is still running, and this is how
        a caller follows it rather than blocking again.
        """
        session = self._require(session_id, owner)
        session.last_used = time.time()
        return clean_output(session.drain())

    def peek(self, session_id: str, *, owner: str = "default") -> str:
        """The retained buffer, without consuming it."""
        session = self._require(session_id, owner)
        return clean_output(session.peek())

    # -- inspection --------------------------------------------------------

    def list(self, *, owner: str | None = None) -> list[Session]:
        sessions = self._sessions.values()
        if owner is not None:
            sessions = [s for s in sessions if s.owner == owner]  # type: ignore[assignment]
        return sorted(sessions, key=lambda s: s.created_at)

    def reap(self) -> int:
        """Forget dead and long-idle sessions. Returns how many went."""
        cutoff = time.time() - IDLE_SESSION_S
        gone = [
            session_id
            for session_id, session in self._sessions.items()
            if not session.alive or session.last_used < cutoff
        ]
        for session_id in gone:
            self._sessions.pop(session_id, None)
        return len(gone)

    def _require(self, session_id: str, owner: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise TerminalFailure(TerminalError.NO_SESSION, f"no session {session_id!r}")
        if session.owner != owner:
            # Deliberately the same message as an unknown session would give
            # if it leaked existence — but a distinct code, because the UI
            # needs to tell a stale id from someone else's.
            raise TerminalFailure(
                TerminalError.FOREIGN_SESSION, f"session {session_id!r} belongs to another owner"
            )
        return session
