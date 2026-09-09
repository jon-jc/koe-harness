"""Persistent terminal sessions.

The unit is a session rather than a command, because the work people do in a
terminal is stateful -- see :mod:`koe.terminal.session` for the ownership,
settlement and buffering rules, and for what a pipe-backed terminal
deliberately cannot do.
"""

from koe.terminal.plugin import terminal_plugin
from koe.terminal.session import (
    BUFFER_BYTES,
    IDLE_MS,
    SEND_TIMEOUT_S,
    Backend,
    SendOutcome,
    Session,
    ShellBackend,
    TerminalError,
    TerminalFailure,
    TerminalService,
    WaitReason,
)

__all__ = [
    "BUFFER_BYTES",
    "IDLE_MS",
    "SEND_TIMEOUT_S",
    "Backend",
    "SendOutcome",
    "Session",
    "ShellBackend",
    "TerminalError",
    "TerminalFailure",
    "TerminalService",
    "WaitReason",
    "terminal_plugin",
]
