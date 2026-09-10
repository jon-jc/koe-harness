"""The agent loop: a turn, its steps, and the tools it calls.

The vocabulary is dsh's, because the distinction it draws is the one that
matters. A **step** is one model request plus the tools that request asked for.
A **turn** is however many steps it takes to owe nothing further: the model
asks to read a file, the file comes back, it asks for another, and only when it
answers without asking for anything is the turn over.

Getting that boundary wrong is the classic agent bug. Treating one request as
the unit means tool results are never seen by the model that asked for them;
letting the loop run unbounded means a model that keeps calling a failing tool
runs until the budget is gone. So a turn is bounded by steps, and hitting that
bound is a reported outcome rather than a silent stop.

**Every step is an event before it is a result.** The loop emits as it goes —
step started, tool called, tool returned, text produced — so a UI can show work
in progress rather than a spinner, and a plugin can audit or intervene without
this module knowing it exists. A loop that only returns at the end can only be
watched by waiting.

**Tool failures are turns, not ends.** A tool that fails hands the model the
failure and lets it try something else, because that is what a person would do.
The alternative — abandoning the turn — throws away everything established so
far to punish a wrong path.

**Vendors are adapters, not branches.** Anthropic and OpenAI disagree about
almost everything in tool calling: the message shape, where results go, what a
stop reason is called. Each gets an adapter that translates to the vocabulary
here, and the loop never learns which one it is talking to. The mock adapter is
the same shape, which is what lets the whole surface be demonstrated with no
key at all.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from koe.text.thinking import strip_thinking
from koe.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

#: How many model requests one turn may make. Reached only by a model that
#: keeps calling tools; a bound rather than a target.
MAX_STEPS = 12

DEFAULT_SYSTEM = """You are the assistant inside koe, a bilingual (Japanese and English) \
voice AI harness that records meetings, transcribes them with speaker labels, and \
generates verified 議事録.

You have tools for reading this workspace's files, running shell commands, and \
reading the transcript and minutes of the meeting recorded in this session. Prefer \
using a tool over guessing: if a question is about the code or the meeting, look.

Answer in the language the user writes in. Be concise and concrete. When you have \
used a tool, say what you found rather than narrating that you used it."""


@dataclass(slots=True)
class ToolCall:
    """A model's request to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelReply:
    """One model response, in the vocabulary the loop uses.

    Whatever the vendor called these fields, an adapter has already turned
    them into this.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class ChatMessage:
    """One entry in the conversation, as the loop stores it.

    `tool_calls` and `tool_results` sit beside the text rather than being
    encoded into it, so an adapter can render them the way its vendor wants
    and a UI can show them as structure instead of parsing prose.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "at": self.at,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ],
            "tool_results": [r.to_dict() for r in self.tool_results],
        }


class Adapter:
    """What the loop needs from a vendor.

    A class rather than a Protocol so `name` has somewhere to live and the
    mock can subclass without restating the interface.
    """

    name = "adapter"

    async def reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
    ) -> ModelReply:
        raise NotImplementedError


@dataclass(slots=True)
class TurnResult:
    """What one turn produced."""

    text: str
    steps: int
    tool_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    stopped_at_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": round(self.duration_ms, 1),
            "stopped_at_limit": self.stopped_at_limit,
        }


#: Called with (event_name, payload) as the turn proceeds.
Listener = Callable[[str, dict[str, Any]], Awaitable[None]]


class Conversation:
    """A chat session: its history, and the loop that advances it."""

    def __init__(
        self,
        adapter: Adapter,
        tools: ToolRegistry,
        *,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = MAX_STEPS,
        owner: str = "chat",
    ) -> None:
        self.id = f"c_{uuid.uuid4().hex[:12]}"
        self.adapter = adapter
        self.tools = tools
        self.system = system
        self.max_steps = max_steps
        self.owner = owner
        self.messages: list[ChatMessage] = []

    # -- history -----------------------------------------------------------

    def history(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def clear(self) -> None:
        self.messages.clear()

    # -- the loop ----------------------------------------------------------

    async def send(self, text: str, *, on_event: Listener | None = None) -> TurnResult:
        """Run one turn: steps until the model owes nothing further."""
        started = time.perf_counter()
        self.messages.append(ChatMessage(role="user", content=text))

        async def emit(event: str, payload: dict[str, Any]) -> None:
            if on_event is not None:
                await on_event(event, payload)

        await emit("turn/start", {"conversation": self.id})

        answer = ""
        steps = 0
        calls = 0
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        hit_limit = False

        while steps < self.max_steps:
            steps += 1
            await emit("step/start", {"step": steps})

            try:
                reply = await self.adapter.reply(
                    self.messages, system=self.system, tools=self.tools.schemas()
                )
            except Exception as exc:
                logger.exception("model request failed")
                await emit("turn/error", {"message": f"{type(exc).__name__}: {exc}"})
                answer = f"The model request failed: {exc}"
                break

            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens
            cost += reply.cost_usd

            # Reasoning models return their scratchpad in <think> tags. Strip it
            # once, here, rather than at each of the three places it otherwise
            # leaks: the event a UI renders, the answer this turn returns, and
            # the message history -- where it would also be re-sent as context
            # on every following turn, paid for again each time.
            text = strip_thinking(reply.text)

            if text:
                answer = text
                await emit("assistant/text", {"text": text, "step": steps})

            if not reply.wants_tools:
                self.messages.append(ChatMessage(role="assistant", content=text))
                break

            self.messages.append(
                ChatMessage(role="assistant", content=text, tool_calls=reply.tool_calls)
            )

            results: list[ToolResult] = []
            for call in reply.tool_calls:
                calls += 1
                await emit(
                    "tool/call",
                    {"id": call.id, "name": call.name, "arguments": call.arguments},
                )
                result = await self.tools.call(
                    call.name, call.arguments, call_id=call.id, owner=self.owner
                )
                results.append(result)
                await emit("tool/result", result.to_dict())

            # Results go back as one message: a vendor expects every call in a
            # response to be answered together, and splitting them produces a
            # request the vendor rejects rather than one it partially answers.
            self.messages.append(ChatMessage(role="tool", tool_results=results))
        else:
            # The `while` ran out rather than breaking: the model was still
            # asking for tools at the step limit.
            hit_limit = True
            answer = answer or (
                f"Stopped after {self.max_steps} steps — the model was still calling tools."
            )
            await emit("turn/limit", {"steps": steps})

        turn = TurnResult(
            text=answer,
            steps=steps,
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            stopped_at_limit=hit_limit,
        )
        await emit("turn/end", turn.to_dict())
        return turn


def render_tool_results(results: Sequence[ToolResult]) -> str:
    """Tool results as text, for an adapter whose vendor has no result role."""
    return "\n\n".join(f"<{r.tool}>\n{r.content}\n</{r.tool}>" for r in results)


def compact_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    """A one-line rendering of a call's arguments, for logs and the UI."""
    try:
        text = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(arguments)
    return text if len(text) <= limit else text[: limit - 1] + "…"
