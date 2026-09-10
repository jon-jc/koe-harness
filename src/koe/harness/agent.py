"""The turn/step machine.

Ported from DeepSeek Harness's `ReactLoopAgent` (MIT). The vocabulary is dsh's
because the distinction it draws is the one that matters, and koe already used
it for the simple loop this replaces:

* A **step** is one model request plus the tools that request asked for.
* A **turn** is however many steps it takes to owe nothing further.

What the port adds over koe's previous loop is everything around that pair.

**The driver is a queue consumer, not a function call.** `send()` queues and
wakes; it does not run a turn. So a message arriving mid-turn is not a
concurrency problem to be locked out, it is input arriving at a queue -- which
is what makes steering possible at all. `kick()` runs turns until the queue is
empty and then goes idle.

**Cancellation is explicit and cooperative.** `cancel()` aborts the current
activity; it does not unwind the log. Text already streamed to the user is
committed as an interrupted assistant message, because the next request must
contain what the user actually saw. A cancelled turn that silently discarded
its output would leave the model and the user with different histories, and the
model would be the one that is wrong.

**Every boundary is a fact.** `turn/start`, `step/start`, `step/end`,
`turn/end` with a reason. The turn's outcome is recorded even when the outcome
is a failure, which is what lets the harness view show a turn that ended badly
rather than a turn that appears never to have finished.

**Failures end the turn, not the loop.** A tool that throws, a provider that
refuses -- the turn closes with an error reason and the agent returns to idle
ready for the next one. dsh's phrase for it is that plugin failure ends the
turn; koe keeps that, because a chat surface that dies on one bad request is a
chat surface nobody trusts with the next one.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from koe.agent.loop import Adapter, ToolCall
from koe.harness.commands import CommandRegistry, CommandResult, builtin_commands
from koe.harness.compaction import Compactor, ModelSummarizer
from koe.harness.inbox import NEXT_STEP, NEXT_TURN, Inbox, InboxTarget, Pending
from koe.harness.prompt import SystemPrompt
from koe.harness.scheduler import (
    DEFAULT_MAX_PARALLEL,
    execute_tool_calls,
    parse_arguments,
)
from koe.harness.session import (
    ASSISTANT_MESSAGE,
    REQUEST_HEADER,
    STEP_END,
    STEP_START,
    TURN_END,
    TURN_START,
    USER_MESSAGE,
    Session,
)
from koe.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: Steps one turn may take before the loop stops on its own. A bound, not a
#: target: only a model that keeps calling tools reaches it.
MAX_STEPS = 12

#: Why a turn ended. `completed` is the model answering without asking for
#: anything further; the rest are the ways that does not happen.
TurnReason = Literal["completed", "cancelled", "error", "max-steps", "concluded", "blocked"]

#: Notified as the turn proceeds. Same shape as koe's existing chat listener,
#: so the websocket surface did not have to change to gain any of this.
Listener = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class TurnOutcome:
    """What one turn produced."""

    text: str = ""
    turn: int = 0
    steps: int = 0
    tool_calls: int = 0
    reason: TurnReason = "completed"
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "turn": self.turn,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "reason": self.reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": round(self.duration_ms, 1),
            # Kept for the callers that predate the harness and read this name.
            "stopped_at_limit": self.reason == "max-steps",
        }


class HarnessAgent:
    """One agent: a session, an inbox, and the driver that consumes it."""

    def __init__(
        self,
        adapter: Adapter,
        tools: ToolRegistry,
        *,
        system: str = "",
        session: Session | None = None,
        max_steps: int = MAX_STEPS,
        max_parallel_tool_calls: int = DEFAULT_MAX_PARALLEL,
        owner: str = "chat",
        on_event: Listener | None = None,
        compactor: Compactor | None = None,
        prompt: SystemPrompt | None = None,
        commands: CommandRegistry | None = None,
    ) -> None:
        self.id = f"a_{uuid.uuid4().hex[:12]}"
        self.adapter = adapter
        self.tools = tools
        self.session = session or Session(system=system)
        self.inbox = Inbox(self.session)
        self.max_steps = max_steps
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.owner = owner
        self.on_event = on_event
        #: Keeps the conversation inside the context window. Its own object
        #: so a deployment can retune the thresholds, or swap the estimator
        #: for a real tokenizer, without touching the loop.
        self.compactor = compactor or Compactor()
        #: Assembled per request from whatever is mounted, so the prompt
        #: describes the harness that exists rather than the one someone
        #: wrote about once.
        self.prompt = prompt
        self.commands = commands or builtin_commands()

        self._phase: Literal["idle", "running"] = "idle"
        self._turn = 0
        self._cancelled = asyncio.Event()
        self._cancel_reason: str = ""
        self._idle = asyncio.Event()
        self._idle.set()
        self._driver: asyncio.Task[None] | None = None
        self._outcomes: list[TurnOutcome] = []

    # -- status -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._phase == "running"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "agent": self.id,
            "session": self.session.id,
            "phase": self._phase,
            "turn": self._turn,
            "pending": self.inbox.to_dict(),
        }

    async def wait_idle(self) -> None:
        """Block until the driver has nothing left to do."""
        await self._idle.wait()

    # -- input ------------------------------------------------------------

    def send(self, text: str, target: InboxTarget = NEXT_TURN) -> Pending:
        """Queue input and wake the driver.

        Queue-then-wake, never run-directly. The caller does not wait for a
        turn, which is what lets the same method be used from a websocket
        handler while a turn is already in flight.
        """
        message = self.inbox.send(Pending(text=text), target)
        self._wake()
        return message

    def followup(self, text: str) -> Pending:
        """Ask for something, in its own turn."""
        return self.send(text, NEXT_TURN)

    def steer(self, text: str) -> Pending:
        """Correct the turn that is already running, at its next boundary."""
        return self.send(text, NEXT_STEP)

    def inject(self, text: str) -> Pending:
        """Add harness-authored context to the running turn."""
        message = self.inbox.inject(text)
        self._wake()
        return message

    def cancel(self, reason: str = "user", *, keep_inbox: bool = False) -> None:
        """Stop the current turn.

        `keep_inbox` is the difference between "stop doing that" and "stop
        entirely". A user who cancels one answer usually still wants the three
        things they queued behind it.
        """
        self._cancel_reason = reason
        self._cancelled.set()
        if not keep_inbox:
            self.inbox.clear()

    def _wake(self) -> None:
        """Start the driver if it is not already running."""
        if self._driver is None or self._driver.done():
            self._idle.clear()
            self._driver = asyncio.ensure_future(self._kick())

    # -- the driver -------------------------------------------------------

    async def _kick(self) -> None:
        """Run turns until the queue is empty.

        Failures are contained here: a turn that ends badly has already
        recorded why, and the driver's job is to return the agent to idle so the
        next message is answerable.
        """
        self._phase = "running"
        try:
            while await self._turn_once():
                pass
        except Exception:
            logger.exception("agent %s: driver failed", self.id)
        finally:
            self._phase = "idle"
            self._idle.set()
            if self.inbox.pending:
                # Something arrived while the driver was shutting down.
                self._wake()

    async def _turn_once(self) -> bool:
        """One turn. Returns whether another should follow."""
        self._cancelled.clear()
        self._turn += 1
        turn = self._turn
        started = time.perf_counter()

        self.session.append(TURN_START, {"turn": turn})
        await self._emit("turn/start", {"turn": turn})

        outcome = TurnOutcome(turn=turn)
        target: InboxTarget = NEXT_TURN
        step = 0

        try:
            while step < self.max_steps:
                claimed = self.inbox.claim(target, turn)

                if step == 0 and not claimed:
                    # Woken with nothing to do: the queue was cleared between
                    # the wake and the claim. The turn owns the boundary and
                    # spends no model call.
                    outcome.reason = "completed"
                    break

                for message in claimed:
                    self.session.append(
                        USER_MESSAGE, {"turn": turn, "text": message.text, "kind": message.kind}
                    )
                    await self._emit(
                        "user/message", {"turn": turn, "text": message.text, "kind": message.kind}
                    )

                step += 1
                self.session.append(STEP_START, {"turn": turn, "step": step})
                await self._emit("step/start", {"turn": turn, "step": step})
                try:
                    wants_more = await self._step(turn, step, outcome)
                finally:
                    self.session.append(STEP_END, {"turn": turn, "step": step})
                    await self._emit("step/end", {"turn": turn, "step": step})

                outcome.steps = step

                if self._cancelled.is_set():
                    outcome.reason = "cancelled"
                    break
                if not wants_more:
                    # The model owes nothing further. Steering that arrived
                    # during the step still gets a step, because the user said
                    # it before seeing the answer.
                    if self.inbox.next_step:
                        target = NEXT_STEP
                        continue
                    break
                target = NEXT_STEP
            else:
                outcome.reason = "max-steps"
        except Exception as exc:
            logger.exception("agent %s: turn %d failed", self.id, turn)
            outcome.reason = "error"
            outcome.text = outcome.text or f"The turn failed: {exc}"
            await self._emit("turn/error", {"turn": turn, "message": str(exc)})
        finally:
            outcome.duration_ms = (time.perf_counter() - started) * 1000.0
            self.session.append(TURN_END, {"turn": turn, "reason": outcome.reason})
            await self._emit("turn/end", {"turn": turn, **outcome.to_dict()})
            self._outcomes.append(outcome)

        # Another turn only if something is waiting for one. A fresh
        # cancellation flag, because the one that ended this turn must not end
        # the next.
        return bool(self.inbox.next_turn) and not self._cancelled.is_set()

    async def _step(self, turn: int, step: int, outcome: TurnOutcome) -> bool:
        """One model request and its tools. Returns whether to take another."""
        schemas = self.tools.schemas()
        await self._relieve_pressure(turn, step, schemas)

        history = self.session.derive_messages()
        self._log_header(turn, step, schemas)

        reply = await self.adapter.reply(history, system=self._system_text(), tools=schemas)
        outcome.input_tokens += reply.input_tokens
        outcome.output_tokens += reply.output_tokens

        from koe.text.thinking import strip_thinking

        text = strip_thinking(reply.text)
        interrupted = self._cancelled.is_set()

        # Committed before the tools run, and committed even when cancelled:
        # the next request has to contain what the user was shown.
        self.session.append(
            ASSISTANT_MESSAGE,
            {
                "turn": turn,
                "step": step,
                "text": text,
                "interrupted": interrupted,
                "calls": [call.name for call in reply.tool_calls],
            },
        )
        if text:
            outcome.text = text
            await self._emit("assistant/text", {"turn": turn, "step": step, "text": text})

        if interrupted or not reply.tool_calls:
            return False

        calls = [
            ToolCall(id=call.id, name=call.name, arguments=parse_arguments(call.arguments))
            for call in reply.tool_calls
        ]
        for call in calls:
            await self._emit(
                "tool/call", {"turn": turn, "step": step, "name": call.name, "id": call.id}
            )

        batch = await execute_tool_calls(
            self.tools,
            self.session,
            calls,
            turn=turn,
            step=step,
            owner=self.owner,
            cancelled=self._cancelled,
            max_parallel=self.max_parallel_tool_calls,
            accept_context=self._accept_context,
        )
        outcome.tool_calls += len(batch.results)
        for result in batch.results:
            await self._emit(
                "tool/result",
                {
                    "turn": turn,
                    "step": step,
                    "name": result.tool,
                    "id": result.call_id,
                    "ok": result.ok,
                    "content": result.content[:2000],
                    "error": result.error.value if result.error else None,
                },
            )

        if batch.concluded:
            outcome.reason = "concluded"
            return False
        return not batch.aborted

    def _system_text(self) -> str:
        """The prompt for this request.

        Assembled from the mounted sections when a registry is present, so
        turning a plugin off takes its instructions with it. Falls back to the
        session's own prompt otherwise, which is what a session created without
        a registry -- a test, a script -- should get.

        An assembly failure falls back rather than ending the turn: a prompt
        that is merely stale still answers the question, and a turn that dies
        because a section could not resolve a variable answers nothing.
        """
        if self.prompt is None:
            return self.session.system_prompt()
        try:
            return self.prompt.render()
        except Exception:
            logger.exception("agent %s: prompt assembly failed", self.id)
            return self.session.system_prompt()

    async def command(self, text: str) -> CommandResult | None:
        """Run `text` as a slash command, or None when it is not one.

        Commands never reach the inbox: they are instructions to the harness,
        not things the user said to the model, and letting one into history
        would have the model reading `/compact` as a request to talk about
        compacting.
        """
        return await self.commands.dispatch(self, text)

    async def _relieve_pressure(self, turn: int, step: int, schemas: list[dict[str, Any]]) -> None:
        """Compact if the surface has grown into the context window.

        At the step boundary, before the request is assembled, because that is
        the last moment the decision can still change what gets sent. Pruning
        is tried first: it costs nothing and often removes the pressure on its
        own, and only if it does not is a summarizing request spent.

        A failure here is logged and dropped. Compaction is what keeps a long
        conversation possible; it is not worth ending a turn over, and the
        request that follows will either fit or be refused by the provider with
        a message the user can act on.
        """
        try:
            measurement = self.compactor.meter.measure(self.session, schemas=schemas)
            if not self.compactor.should_compact(measurement):
                return

            pruned = self.compactor.prune(self.session)
            if pruned.ok:
                await self._emit("compaction", {"turn": turn, "step": step, **pruned.to_dict()})
                measurement = self.compactor.meter.measure(self.session, schemas=schemas)
                if not self.compactor.should_compact(measurement):
                    return

            result = await self.compactor.compact(
                self.session, ModelSummarizer(self.adapter), schemas=schemas
            )
            if result.ok or result.error:
                await self._emit("compaction", {"turn": turn, "step": step, **result.to_dict()})
        except Exception:
            logger.exception("agent %s: compaction failed", self.id)

    def _accept_context(self, text: str) -> None:
        """Stage a tool's extra context for the next step.

        The scheduler cannot queue it itself: staging belongs to the agent,
        which owns the inbox, and a scheduler that could write to the queue
        would be able to extend the turn it was asked to finish.
        """
        self.inbox.inject(text)

    def _log_header(self, turn: int, step: int, schemas: list[dict[str, Any]]) -> None:
        """Record the request envelope, but only when it changed.

        Config and tool schemas, never the prompt. An unchanged envelope
        inherits the previous header, which keeps the log readable -- a
        twelve-step turn writes one header, not twelve -- and matches what
        actually varies between steps.
        """
        header = {
            "provider": getattr(self.adapter, "name", "adapter"),
            "model": getattr(getattr(self.adapter, "_provider", None), "model", ""),
            "tools": sorted(schema["name"] for schema in schemas),
        }
        previous = self.session.request_header()
        if (
            previous is not None
            and {key: previous.get(key) for key in ("provider", "model", "tools")} == header
        ):
            return
        self.session.append(REQUEST_HEADER, {"turn": turn, "step": step, **header})

    async def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(event, payload)
        except Exception:
            # A listener that fails must not end a turn. The socket it was
            # writing to has usually just closed.
            logger.debug("agent %s: listener failed on %s", self.id, event, exc_info=True)

    # -- convenience ------------------------------------------------------

    async def ask(self, text: str) -> TurnOutcome:
        """Send one message and wait for the turn it produces.

        The blocking façade over a non-blocking driver, for callers that want
        one answer -- a script, the non-streaming endpoint -- rather than a
        conversation they steer.
        """
        before = len(self._outcomes)
        self.followup(text)
        # Wait for the driver to reach idle *after* producing at least this
        # turn: waiting on idle alone races a driver that has not started.
        while len(self._outcomes) <= before:
            await asyncio.sleep(0)
            if self._driver is not None and self._driver.done() and len(self._outcomes) <= before:
                break
            await self.wait_idle()
        return self._outcomes[-1] if len(self._outcomes) > before else TurnOutcome(reason="blocked")

    def history(self) -> list[dict[str, Any]]:
        """The derived message history, as the chat UI reads it."""
        return [message.to_dict() for message in self.session.derive_messages()]

    def clear(self) -> None:
        """Start over with an empty log, keeping the system prompt."""
        system = self.session.system
        self.session.close()
        self.session = Session(system=system)
        self.inbox = Inbox(self.session)
        self._turn = 0
        self._outcomes.clear()


@dataclass(slots=True)
class AgentHandle:
    """An agent plus the disposer that owns its teardown.

    Handing back a disposer rather than exposing `remove(id)` is the same
    argument the tool registry makes: whoever created the agent holds the only
    thing that removes it, so a caller cannot tear down an agent it does not
    own, and a stale handle cannot remove a later agent that reused the id.
    """

    agent: HarnessAgent
    dispose: Callable[[], None]


class AgentRegistry:
    """Live agents, keyed by id.

    Small on purpose. dsh's registry carries delegation, persistence handles
    and resume; koe's chat needs create, look up, and dispose, and the parts
    that are not here are absent rather than stubbed.
    """

    def __init__(self, *, limit: int = 32) -> None:
        self._agents: dict[str, HarnessAgent] = {}
        self._limit = limit

    def create(self, agent: HarnessAgent) -> AgentHandle:
        self._agents[agent.id] = agent

        def dispose() -> None:
            # Bound to this exact object, so a late disposer cannot evict a
            # newer agent that happens to share the id.
            if self._agents.get(agent.id) is agent:
                del self._agents[agent.id]
            agent.cancel("disposed")
            agent.session.close()

        while len(self._agents) > self._limit:
            oldest = next(iter(self._agents))
            self._agents.pop(oldest).cancel("evicted")
        return AgentHandle(agent=agent, dispose=dispose)

    def get(self, agent_id: str) -> HarnessAgent | None:
        return self._agents.get(agent_id)

    def by_session(self, session_id: str) -> HarnessAgent | None:
        return next(
            (agent for agent in self._agents.values() if agent.session.id == session_id), None
        )

    def __len__(self) -> int:
        return len(self._agents)

    def to_dict(self) -> dict[str, Any]:
        return {"agents": [agent.status for agent in self._agents.values()]}
