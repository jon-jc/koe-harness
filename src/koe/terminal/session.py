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

## Two backends, for two different readers

``shell`` is pipes, and it is the one a model talks to: no escape sequences,
a prompt sentinel we control, output that costs what it says it costs.

``pty`` is a real terminal — ConPTY on Windows, ``pty`` elsewhere — and it is
the one a person talks to. ``vim``, ``top`` and ``less` check ``isatty``, ask
for a window size, and address the cursor; pipes make them degrade or refuse.
Its output is raw: escape sequences are the payload, not noise, so nothing is
stripped and no sentinel is injected. That output is only useful to something
that can render it, which is why the pty path exists *alongside* an emulator
in the client rather than instead of the pipe path.

Neither is a fallback for the other. A model handed a pty pays for a screenful
of ANSI it has to reason about; a person handed pipes cannot run an editor.
"""

from __future__ import annotations

import asyncio
import contextlib
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

from koe.terminal.channels import (
    Channel,
    PipeChannel,
    open_pty,
    pty_available,
)

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

#: Sent to the shell before anything else.
#:
#: Exporting PS1 into the environment is not enough, and assuming it was cost
#: a bug: an interactive bash *assigns* its own default PS1 at startup,
#: overriding what it inherited. The sentinel then never appeared, every send
#: fell back to settling on silence, and `bash-5.3#` leaked into the output as
#: the visible symptom. Assigning it as a command runs after that default is
#: set, which is the only ordering that wins.
INIT_COMMAND = f"PS1='{PROMPT_SENTINEL}\\n'; PS2=''; unset PROMPT_COMMAND\n"


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
    """Opens the channel behind a session.

    A protocol rather than a base class so a container backend or a remote one
    can be dropped in without touching the service — which is exactly what the
    pty backend below did.
    """

    #: The stable name callers open sessions by.
    type: str

    #: True when output is the payload rather than something to clean. A pty
    #: session keeps its escape sequences and gets no prompt sentinel; a pipe
    #: session has both stripped.
    raw: bool

    async def open(self, cwd: Path, size: tuple[int, int]) -> Channel: ...


class ShellBackend:
    """The local shell, over pipes. The backend a model talks to."""

    type = "shell"
    raw = False

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or self._default_command()

    @staticmethod
    def _default_command() -> list[str]:
        """The shell to run here.

        On Windows, prefer Git's bash: the rest of koe's tooling assumes POSIX
        utilities, and a user who has git installed has them.

        The install paths are checked *before* `which`, because `bash.exe` on
        PATH is frequently the WSL launcher in System32. That starts a shell
        in a different filesystem namespace, where the workspace directory we
        hand it is spelled `/mnt/c/...` rather than `/c/...` — so the terminal
        works but is quietly rooted somewhere other than the workspace every
        other surface is showing.

        `cmd.exe` is the fallback that always exists.
        """
        if sys.platform == "win32":
            # Uppercase: Windows environment lookups are case-insensitive, and
            # the convention keeps the linter and the reader in agreement.
            roots = [
                os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"),
            ]
            for tail in (("bin",), ("usr", "bin")):
                for root in roots:
                    candidate = Path(root, "Git", *tail, "bash.exe")
                    if candidate.is_file():
                        return [str(candidate), "--norc", "--noprofile", "-i"]

            found = shutil.which("bash.exe") or shutil.which("bash")
            if found and "system32" not in found.lower():
                return [found, "--norc", "--noprofile", "-i"]
            return [os.environ.get("COMSPEC", "cmd.exe")]
        return [os.environ.get("SHELL", "/bin/bash"), "--norc", "--noprofile", "-i"]

    async def open(self, cwd: Path, size: tuple[int, int] = (80, 24)) -> Channel:
        env = dict(os.environ)
        # A prompt we can recognise, and no colour: escape sequences in a
        # buffer that is going to a model are noise it pays for by the token.
        # Set here *and* sent as INIT_COMMAND after spawn: bash overrides this
        # for an interactive shell, but a shell that is not bash may honour it.
        env["PS1"] = f"{PROMPT_SENTINEL}\n"
        env["PS2"] = ""
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        process = await asyncio.create_subprocess_exec(
            *self._command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        return PipeChannel(process)


class PtyBackend:
    """A real terminal. The backend a person talks to.

    Deliberately keeps the user's rc files, where the shell backend discards
    them: a person opening a terminal wants their aliases and their prompt,
    and there is no sentinel here whose survival depends on suppressing them.
    """

    type = "pty"
    raw = True

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or self._default_command()

    @staticmethod
    def _default_command() -> list[str]:
        """An interactive login shell, with the user's own configuration."""
        if sys.platform == "win32":
            base = ShellBackend._default_command()
            # Drop --norc/--noprofile: the shell backend needs them so its
            # prompt sentinel survives, and a human terminal does not.
            return [base[0], "-i"] if base[0].lower().endswith("bash.exe") else base
        return [os.environ.get("SHELL", "/bin/bash"), "-i"]

    async def open(self, cwd: Path, size: tuple[int, int] = (80, 24)) -> Channel:
        if not pty_available():
            raise OSError(
                "no pty on this machine — install pywinpty on Windows for "
                "full-screen programs, or use the shell backend"
            )
        env = dict(os.environ)
        # A real terminal, so programs that check will draw rather than refuse.
        env["TERM"] = env.get("TERM") or "xterm-256color"
        env.pop("NO_COLOR", None)
        return await open_pty(self._command, str(cwd), env, size)


@dataclass
class Session:
    """One live terminal."""

    id: str
    owner: str
    type: str
    name: str
    cwd: str
    channel: Channel
    #: True when output is the payload: a pty session is not cleaned and
    #: gets no sentinel, because its escape sequences are the content.
    raw: bool = False
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
        return self.channel.alive

    @property
    def exit_code(self) -> int | None:
        return self.channel.exit_code

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

    # `wait_s` rather than `timeout`: the linter reserves that name for a
    # parameter forwarded to `asyncio.timeout`, and this one bounds a single
    # wait rather than the whole call.
    async def follow(self, *, wait_s: float = 30.0) -> str:
        """Wait for output and return it, or "" when the session ends.

        A streaming consumer waits here rather than reading the channel
        itself, because there is exactly one reader — the service's pump — and
        a second one silently steals chunks from the first. That bug is
        invisible in a test that only checks the connection works: output
        still arrives, just not all of it, and the half that went to the other
        reader is simply never seen.
        """
        pending = self.drain()
        if pending:
            return pending
        self._activity.clear()
        try:
            await asyncio.wait_for(self._activity.wait(), timeout=wait_s)
        except TimeoutError:
            return ""
        return self.drain()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "type": self.type,
            "name": self.name,
            "cwd": self.cwd,
            "pid": self.channel.pid,
            "raw": self.raw,
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

    async def open(
        self,
        *,
        owner: str = "default",
        type: str = "shell",
        name: str = "",
        size: tuple[int, int] = (80, 24),
    ) -> Session:
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
            channel = await backend.open(self._cwd, size)
        except (OSError, ValueError, ImportError) as exc:
            raise TerminalFailure(
                TerminalError.SPAWN_FAILED, f"could not start a {type} session: {exc}"
            ) from exc

        session = Session(
            id=f"t_{uuid.uuid4().hex[:12]}",
            owner=owner,
            type=type,
            name=name or f"session {len(self._sessions) + 1}",
            cwd=str(self._cwd),
            channel=channel,
            raw=getattr(backend, "raw", False),
        )
        session._reader = asyncio.create_task(self._pump(session))
        self._sessions[session.id] = session

        # A raw session is handed straight to a terminal emulator, which
        # wants the banner, the prompt and every escape the shell emits. Only
        # the pipe path installs a sentinel and settles before returning.
        if not session.raw:
            # Before the banner wait, so the first prompt anyone sees is
            # already the sentinel rather than whatever the shell chose.
            with contextlib.suppress(OSError, ConnectionError, BrokenPipeError):
                await channel.write(INIT_COMMAND.encode("utf-8"))

            # Settle before returning, so the caller gets a quiet terminal
            # rather than a banner arriving mid-command.
            #
            # Draining once on the first sentinel is not enough, and getting
            # that wrong desynchronized the buffer by a whole command: the
            # shell echoes the init line, so the sentinel can appear while
            # that echo is still in flight, and whatever is left lands on the
            # *next* send's output. Waiting for silence after the prompt is
            # what makes the drain total.
            await self._quiet(session, idle_ms=IDLE_MS, budget_s=5.0)
            await asyncio.sleep(IDLE_GRACE_MS / 1000.0 / 5)
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
        try:
            while True:
                chunk = await session.channel.read()
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
            await asyncio.gather(session._reader, return_exceptions=True)

        # Close the write side before signalling. A shell reading a closed
        # stdin exits on its own, which is a cleaner end than a signal — and
        # leaving the pipe open means asyncio finalizes the transport during
        # garbage collection instead, after the loop has gone, which surfaces
        # as "Event loop is closed" from a destructor nobody can catch.
        await session.channel.aclose()

        if session.alive:
            session.channel.terminate()
            try:
                await asyncio.wait_for(session.channel.wait(), timeout=5.0)
            except TimeoutError:
                # A shell ignoring SIGTERM is usually one with a foreground
                # child that ignored it too. Killing the shell is what we can
                # reach from here.
                session.channel.kill()
                await session.channel.wait()

        # Give the proactor a turn to finish tearing the pipes down, so the
        # transports are collected inside the loop's lifetime rather than
        # after it.
        await asyncio.sleep(0)

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
        session._sending = True
        session.last_used = time.time()
        started = time.perf_counter()
        try:
            # Drain first: output from before this send belongs to whatever
            # produced it, not to the command about to run.
            session.drain()
            payload = text + ("\n" if submit and not text.endswith("\n") else "")
            await session.channel.write(payload.encode("utf-8"))
            reason = await self._quiet(
                session, idle_ms=IDLE_MS, budget_s=timeout_s, expect_prompt=not session.raw
            )
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
