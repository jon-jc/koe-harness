"""The terminal, as a plugin: the service, a backend, and the model's tools.

Split the way dsh splits it, and for the same reason. The *service* mints
sessions and enforces ownership; a *backend* knows how to spawn a process; the
*tools* are how a model reaches them. Three things rather than one because they
change for different reasons: adding a ConPTY backend should not touch the
tools, and rewording a tool description should not touch the spawning.

Mounting this plugin is what makes a terminal exist. A deployment that should
not have one does not mount it — there is no flag, and no code path that has to
remember to check one.
"""

from __future__ import annotations

import logging
from typing import Any

from koe.terminal.session import (
    ShellBackend,
    TerminalFailure,
    TerminalService,
    WaitReason,
)
from koe.tools.registry import ToolInvocationError, ToolRun, ToolSpec

logger = logging.getLogger(__name__)

#: What each settlement means, in words the model can act on. A bare
#: `inferred_idle` tells it nothing; "still running, read again" tells it what
#: to do next, which is the entire purpose of returning the reason.
_ADVICE: dict[WaitReason, str] = {
    WaitReason.PROMPT: "",
    WaitReason.IDLE: (
        "\n\n[the shell went quiet without returning to its prompt — it is "
        "probably waiting for input, e.g. inside a REPL]"
    ),
    WaitReason.TIMEOUT: (
        "\n\n[still running: the wait budget ran out, not the command. Use "
        "terminal_read to collect more output]"
    ),
    WaitReason.SESSION_EXIT: "\n\n[the shell exited; open a new session to continue]",
}


def terminal_plugin(ctx: Any, config: Any = None) -> None:
    """Provide `ctx.terminals` and the tools that drive it."""
    registry = ctx.get("tools")
    workspace = ctx.get("workspace")

    service = TerminalService(cwd=workspace.root if workspace else None)
    service.register_backend(ShellBackend())
    ctx.provide("terminals", service, replace=True)

    # Unloading this plugin must not leave orphaned shells behind: a leaked
    # process outlives the app that spawned it, and nothing else will reap it.
    ctx.scope.collect("terminal:dispose", lambda: _dispose(service))

    if registry is None:
        return

    async def terminal_run(args: dict[str, Any], run: ToolRun) -> Any:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolInvocationError("'command' is required")

        session_id = args.get("session")
        timeout = float(args.get("timeout_s") or 30.0)
        try:
            if not session_id:
                # The common case is one command with no follow-up, so a
                # missing session opens one rather than making the model
                # perform a ceremony it did not ask for.
                session = await service.open(owner=run.owner, name="agent")
                session_id = session.id
            outcome = await service.send(
                str(session_id), command, owner=run.owner, timeout_s=timeout
            )
        except TerminalFailure as exc:
            raise ToolInvocationError(f"{exc} ({exc.code.value})") from exc

        body = outcome.output or "[no output]"
        return f"session {session_id}\n\n{body}{_ADVICE[outcome.reason]}"

    async def terminal_read(args: dict[str, Any], run: ToolRun) -> Any:
        session_id = args.get("session")
        if not isinstance(session_id, str) or not session_id:
            raise ToolInvocationError("'session' is required")
        try:
            return service.read(session_id, owner=run.owner) or "[nothing new]"
        except TerminalFailure as exc:
            raise ToolInvocationError(f"{exc} ({exc.code.value})") from exc

    async def terminal_list(args: dict[str, Any], run: ToolRun) -> Any:
        sessions = service.list(owner=run.owner)
        if not sessions:
            return "no open sessions"
        return "\n".join(
            f"{s.id}  {s.name}  {'running' if s.alive else 'exited'}  cwd={s.cwd}" for s in sessions
        )

    async def terminal_close(args: dict[str, Any], run: ToolRun) -> Any:
        session_id = args.get("session")
        if not isinstance(session_id, str) or not session_id:
            raise ToolInvocationError("'session' is required")
        try:
            await service.close(session_id, owner=run.owner)
        except TerminalFailure as exc:
            raise ToolInvocationError(f"{exc} ({exc.code.value})") from exc
        return f"closed {session_id}"

    specs = [
        ToolSpec(
            name="terminal_run",
            description=(
                "Run a shell command and return its output. Omit 'session' to "
                "run in a fresh shell; pass the session id returned by an "
                "earlier call to keep working in the same one, which is what "
                "you need when a directory change, an environment variable or "
                "a REPL has to persist. If the result says the command is "
                "still running, call terminal_read rather than sending again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "session": {
                        "type": "string",
                        "description": "Existing session id. Omit to open a new shell.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "How long to wait before returning. Default 30.",
                    },
                },
                "required": ["command"],
            },
            execute=terminal_run,
            # Declared, not enforced here: an approval policy decides what a
            # dangerous tool means for a given deployment.
            dangerous=True,
            timeout_s=120.0,
            source="terminal",
        ),
        ToolSpec(
            name="terminal_read",
            description=(
                "Collect output that has arrived since the last read, without "
                "waiting. Use it after a command reported that it is still "
                "running."
            ),
            parameters={
                "type": "object",
                "properties": {"session": {"type": "string"}},
                "required": ["session"],
            },
            execute=terminal_read,
            source="terminal",
        ),
        ToolSpec(
            name="terminal_list",
            description="List your open terminal sessions and whether each is still running.",
            parameters={"type": "object", "properties": {}},
            execute=terminal_list,
            source="terminal",
        ),
        ToolSpec(
            name="terminal_close",
            description="End a terminal session and its process tree.",
            parameters={
                "type": "object",
                "properties": {"session": {"type": "string"}},
                "required": ["session"],
            },
            execute=terminal_close,
            dangerous=True,
            source="terminal",
        ),
    ]
    for spec in specs:
        ctx.scope.collect(f"tool:{spec.name}", registry.register(spec))


#: Disposal tasks are held here for their lifetime. Without a reference the
#: event loop only keeps a weak one, and a garbage collection between the
#: schedule and the first await cancels the teardown silently.
_PENDING: set[Any] = set()


def _dispose(service: TerminalService) -> None:
    """Close every session when the plugin unloads.

    Teardown is synchronous but disposal is async, so this schedules the work
    on the running loop and settles for best effort when there is no loop —
    which is process exit, where the OS reaps the children anyway.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(service.dispose())
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)
