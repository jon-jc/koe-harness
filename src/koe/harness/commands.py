"""Slash commands: the things a user asks the harness, not the model.

Ported from DeepSeek Harness's command seam and its `command-compact` (MIT).

The distinction is the whole point. `/compact` is not a request for the model
to summarize -- it is an instruction to the *harness* to run a compaction
transaction, and sending it to the model instead would produce a polite reply
about summarizing and no compaction. Likewise `/clear` is not "please forget",
it is a new session.

**A command runs instead of a turn, and says so.** It never reaches the inbox,
so it cannot be mistaken for something the user said to the model, and it does
not appear in derived history.

**Failures are the interesting part**, which is why dsh's `/compact` spends
most of its code on them. "It did not work" is useless to someone who now has
to decide whether their conversation is intact. Each outcome here says what
happened *to the conversation*: unchanged, changed, or recorded in the log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What a command did, in words meant for the person who typed it."""

    text: str
    ok: bool = True
    #: Set when the command changed the session enough that a client showing
    #: history should reload rather than append.
    reload: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "ok": self.ok, "reload": self.reload}


@dataclass(frozen=True, slots=True)
class Command:
    """One command a user can type."""

    name: str
    summary: str
    run: Callable[[Any, str], Awaitable[CommandResult]]
    #: Shown in help. Commands that take no arguments say so, because the
    #: common failure is someone typing `/compact everything` and wondering
    #: which part was ignored.
    usage: str = ""


@dataclass(slots=True)
class CommandRegistry:
    """Commands, and the parser that routes to them."""

    _commands: dict[str, Command] = field(default_factory=dict, init=False, repr=False)

    def register(self, command: Command) -> Callable[[], None]:
        if command.name in self._commands:
            raise ValueError(f"command /{command.name} is already registered")
        self._commands[command.name] = command

        def dispose() -> None:
            if self._commands.get(command.name) is command:
                del self._commands[command.name]

        return dispose

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def __len__(self) -> int:
        return len(self._commands)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": command.name, "summary": command.summary, "usage": command.usage}
            for command in sorted(self._commands.values(), key=lambda c: c.name)
        ]

    @staticmethod
    def parse(text: str) -> tuple[str, str] | None:
        """Split `/name rest` into its parts, or None if this is not a command.

        A leading slash and nothing else. Deliberately not a general parser:
        anything cleverer starts guessing about text that begins with a slash
        because it is a path, and sending a user's message to a command handler
        because it happened to start with `/usr` is worse than having no
        commands.
        """
        stripped = text.strip()
        if not stripped.startswith("/") or len(stripped) < 2:
            return None
        head, _, rest = stripped[1:].partition(" ")
        if not head or not head.replace("-", "").replace("_", "").isalnum():
            return None
        return head.lower(), rest.strip()

    async def dispatch(self, agent: Any, text: str) -> CommandResult | None:
        """Run the command in `text`, or None when it is not one."""
        parsed = self.parse(text)
        if parsed is None:
            return None
        name, rest = parsed

        command = self.get(name)
        if command is None:
            known = ", ".join(f"/{entry}" for entry in sorted(self._commands))
            return CommandResult(
                text=f"Unknown command /{name}. Available: {known or '(none)'}", ok=False
            )
        try:
            return await command.run(agent, rest)
        except Exception as exc:
            logger.exception("command /%s failed", name)
            # Named as unfinished rather than failed: the user needs to know
            # whether their conversation changed, and a command that threw
            # halfway cannot promise it did not.
            return CommandResult(
                text=f"/{name} did not finish: {exc}. Check the session before retrying.",
                ok=False,
            )


# --------------------------------------------------------------------------
# the built-in commands
# --------------------------------------------------------------------------


async def _compact(agent: Any, rest: str) -> CommandResult:
    """Compact now, rather than waiting for the threshold."""
    if rest:
        return CommandResult(text="Usage: /compact (no arguments)", ok=False)
    if agent.running:
        # dsh refuses the same case. A compaction racing a turn would replace
        # the history the turn is deriving its next request from.
        return CommandResult(
            text="The agent is working. Stop the turn first, or wait for it to finish.",
            ok=False,
        )

    from koe.harness.compaction import ModelSummarizer

    before = agent.compactor.meter.measure(agent.session, schemas=agent.tools.schemas())
    pruned = agent.compactor.prune(agent.session)
    result = await agent.compactor.compact(
        agent.session, ModelSummarizer(agent.adapter), schemas=agent.tools.schemas()
    )
    after = agent.compactor.meter.measure(agent.session, schemas=agent.tools.schemas())

    if result.error:
        return CommandResult(
            text=(
                f"Compaction did not run: {result.error}. "
                "The conversation is unchanged; the attempt is in the session log."
            ),
            ok=False,
        )
    if not result.ok and not pruned.ok:
        return CommandResult(
            text="Nothing to compact yet — the conversation still fits comfortably.",
            ok=True,
        )

    saved = before.total - after.total
    parts = []
    if pruned.ok:
        parts.append(f"pruned {len(pruned.shadowed_seqs)} oversized tool results")
    if result.ok:
        parts.append(f"summarized {len(result.shadowed_seqs)} messages")
    return CommandResult(
        text=f"Compacted: {', '.join(parts)}. {saved} tokens freed, nothing removed from the log.",
        reload=True,
    )


async def _clear(agent: Any, rest: str) -> CommandResult:
    """Start a new session, keeping the prompt."""
    if agent.running:
        agent.cancel("cleared")
    agent.clear()
    return CommandResult(text="Started a new conversation.", reload=True)


async def _cancel(agent: Any, rest: str) -> CommandResult:
    if not agent.running:
        return CommandResult(text="Nothing is running.", ok=False)
    agent.cancel("user", keep_inbox=True)
    return CommandResult(text="Stopped. Anything queued behind it is kept.")


async def _context(agent: Any, rest: str) -> CommandResult:
    """How much of the window is in use, and where it went.

    Worth a command of its own: "why is it forgetting things" and "why is this
    costing so much" have the same answer, and it is otherwise invisible.
    """
    measurement = agent.compactor.meter.measure(agent.session, schemas=agent.tools.schemas())
    surface = len(agent.session.surface())
    total = len(agent.session)
    return CommandResult(
        text=(
            f"{measurement.total} of {measurement.context_window} tokens "
            f"({measurement.pressure:.0%}) — {measurement.message_tokens} in messages, "
            f"{measurement.schema_tokens} in tool schemas. "
            f"{surface} of {total} log events are visible to the model."
        )
    )


def _help_for(registry: CommandRegistry) -> Command:
    async def run(agent: Any, rest: str) -> CommandResult:
        lines = [f"/{entry['name']:<9} {entry['summary']}" for entry in registry.describe()]
        return CommandResult(text="\n".join(lines))

    return Command(name="help", summary="List the commands.", run=run)


def builtin_commands() -> CommandRegistry:
    """The registry the chat starts with."""
    registry = CommandRegistry()
    registry.register(
        Command(
            name="compact",
            summary="Summarize the earlier conversation to free context.",
            run=_compact,
            usage="/compact",
        )
    )
    registry.register(
        Command(name="clear", summary="Start a new conversation.", run=_clear, usage="/clear")
    )
    registry.register(
        Command(name="stop", summary="Stop the running turn.", run=_cancel, usage="/stop")
    )
    registry.register(
        Command(
            name="context",
            summary="Show how much of the context window is in use.",
            run=_context,
            usage="/context",
        )
    )
    registry.register(_help_for(registry))
    return registry
