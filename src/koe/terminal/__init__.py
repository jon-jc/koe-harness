"""Persistent terminal sessions.

The unit is a session rather than a command, because the work people do in a
terminal is stateful -- see :mod:`koe.terminal.session` for the ownership,
settlement and buffering rules, and for why there are two backends rather than
one: pipes for a model, a pty for a person.
"""

from koe.terminal.channels import Channel, PipeChannel, pty_available
from koe.terminal.plugin import terminal_plugin
from koe.terminal.session import (
    BUFFER_BYTES,
    IDLE_MS,
    SEND_TIMEOUT_S,
    Backend,
    PtyBackend,
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
    "Channel",
    "PipeChannel",
    "PtyBackend",
    "SendOutcome",
    "Session",
    "ShellBackend",
    "TerminalError",
    "TerminalFailure",
    "TerminalService",
    "WaitReason",
    "pty_available",
    "terminal_plugin",
]
