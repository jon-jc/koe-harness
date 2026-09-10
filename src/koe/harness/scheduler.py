"""Scheduling one step's tool calls.

A close port of DeepSeek Harness's `tool-calls.ts` (MIT). koe used to run tool
calls one after another in the order the model listed them, which is correct and
slow: three independent file reads took three round trips of latency for no
reason.

Four rules, and every one of them is the interesting part.

**Exclusive calls are barriers; parallel-safe calls share a bounded pool.** A
read is parallel-safe. A shell command is not -- it can change the working
directory, write a file another call is reading, or hold a lock. Safety is a
property of the tool, declared on the tool, and the scheduler reads it rather
than guessing from the name.

**Dispatch overlaps, but results commit in model order.** This is the rule that
makes parallelism invisible to the model, and it is the one a naive
implementation gets wrong. If three reads finish out of order and are appended
as they land, the history the next request derives depends on disk timing -- so
the same conversation replays differently, and a provider's prefix cache misses.
`_commit_ready` therefore advances only across a contiguous run of settled
slots: a call that finished early waits for its predecessors.

**Modes are re-read as the pool fills.** A tool registered mid-step -- by a
plugin the model just enabled -- classifies the calls after it, so a parallel
group ends at the first call that is no longer parallel-safe.

**Cancellation still produces a well-formed history.** Started calls drain and
commit. Calls that never dispatched get a synthetic result carrying
`ABORTED_BEFORE_DISPATCH`, because a model request whose assistant turn contains
a call with no result is malformed -- vendors reject it, and the next turn would
fail on the history rather than on the cancellation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from koe.agent.loop import ToolCall
from koe.harness.session import TOOL_CALL, Session, log_tool_result
from koe.tools.registry import ToolError, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

#: Parallel-safe calls in flight at once. dsh's default, and the reasoning
#: carries: it is high enough that a fan-out of reads is one round trip, and low
#: enough that a model asking for fifty things does not open fifty sockets.
DEFAULT_MAX_PARALLEL = 10

#: What a call that never dispatched reports. The text is the model's only
#: explanation, so it says what happened rather than naming a code.
ABORTED_TEXT = "Error: tool call aborted before dispatch"


@dataclass(slots=True)
class ToolBatch:
    """What one step's tool calls produced."""

    results: list[ToolResult] = field(default_factory=list)
    #: Set when cancellation stopped the batch. The turn ends after it.
    aborted: bool = False
    #: Set when a tool asked for the turn to stop -- an interactive prompt, a
    #: handoff. The model gets the result and the loop does not take another
    #: step.
    concluded: bool = False


@dataclass(slots=True)
class _Slot:
    """A dispatched call awaiting its turn to commit."""

    result: ToolResult
    call_seq: int


async def execute_tool_calls(
    tools: ToolRegistry,
    session: Session,
    calls: Sequence[ToolCall],
    *,
    turn: int,
    step: int,
    owner: str,
    cancelled: asyncio.Event,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    accept_context: Callable[[str], None] | None = None,
) -> ToolBatch:
    """Run one step's calls, grouping by execution mode.

    `accept_context` receives text a tool wants added to the next step. The
    scheduler does not queue it directly: staging it belongs to the agent, which
    owns the inbox, and a scheduler that could write to the queue would be able
    to extend the turn it was asked to finish.
    """
    batch = ToolBatch()
    index = 0
    planned = list(calls)

    while index < len(planned):
        # Classified immediately before the group is formed, so a registration
        # that happened during the previous group applies to this one.
        mode = _mode(tools, planned[index])
        group = planned[index:] if mode == "parallel" else planned[index : index + 1]

        outcome = await _run_group(
            tools,
            session,
            group,
            turn=turn,
            step=step,
            owner=owner,
            cancelled=cancelled,
            max_parallel=max_parallel if mode == "parallel" else 1,
            accept_context=accept_context,
            batch=batch,
        )
        index += outcome

        if batch.aborted:
            # Everything the model asked for and cancellation prevented, so the
            # assistant turn has a result for every call it contains.
            for call in planned[index:]:
                _append_aborted(session, call, turn=turn, step=step, batch=batch)
            return batch

    return batch


def _mode(tools: ToolRegistry, call: ToolCall) -> str:
    """How this call may be scheduled. Unknown tools are exclusive.

    Unknown rather than parallel, because an unregistered name is about to
    become an error result and there is nothing to gain by racing it -- and
    because a tool arriving between classification and dispatch is safer
    treated as a barrier.
    """
    spec = tools.get(call.name)
    if spec is None:
        return "exclusive"
    return "parallel" if getattr(spec, "parallel_safe", False) else "exclusive"


async def _run_group(
    tools: ToolRegistry,
    session: Session,
    group: Sequence[ToolCall],
    *,
    turn: int,
    step: int,
    owner: str,
    cancelled: asyncio.Event,
    max_parallel: int,
    accept_context: Callable[[str], None] | None,
    batch: ToolBatch,
) -> int:
    """One barrier or one pool. Returns how many calls it consumed."""
    slots: list[_Slot | None] = [None] * len(group)
    tasks: dict[int, asyncio.Task[int]] = {}
    next_to_start = 0
    committed = 0
    started = 0

    def commit_ready() -> None:
        """Append every settled call whose predecessors have also settled.

        Contiguous, deliberately: a call that finished first still commits
        after the ones the model listed before it, so the derived history does
        not depend on which tool happened to be quicker.
        """
        nonlocal committed
        while committed < len(group):
            slot = slots[committed]
            if slot is None:
                break
            log_tool_result(session, slot.result, turn=turn, step=step, call_seq=slot.call_seq)
            batch.results.append(slot.result)
            if getattr(slot.result, "value", None) and isinstance(slot.result.value, dict):
                extra = slot.result.value.get("additional_context")
                if isinstance(extra, str) and extra and accept_context is not None:
                    accept_context(extra)
                if slot.result.value.get("concludes_turn") is True:
                    batch.concluded = True
            committed += 1

    async def start(position: int) -> int:
        call = group[position]
        event = session.append(
            TOOL_CALL,
            {
                "turn": turn,
                "step": step,
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            },
        )
        result = await tools.call(call.name, call.arguments, call_id=call.id, owner=owner)
        slots[position] = _Slot(result=result, call_seq=event.seq)
        return position

    def fill() -> None:
        """Start as many calls as the pool allows, stopping at a barrier."""
        nonlocal next_to_start, started
        while not cancelled.is_set() and next_to_start < len(group) and len(tasks) < max_parallel:
            if (
                next_to_start > 0
                and max_parallel > 1
                and _mode(tools, group[next_to_start]) != "parallel"
            ):
                # A tool registered since this group formed is not
                # parallel-safe: the group ends here and the caller opens a
                # barrier for it.
                break
            tasks[next_to_start] = asyncio.ensure_future(start(next_to_start))
            next_to_start += 1
            started += 1

    fill()
    while tasks:
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            position = next(key for key, value in tasks.items() if value is task)
            del tasks[position]
            # `tools.call` never raises -- failures come back as results -- so
            # an exception here is a harness bug rather than a tool one, and
            # hiding it would make the next symptom appear somewhere else.
            task.result()
        commit_ready()
        if cancelled.is_set():
            batch.aborted = True
            break
        fill()

    if tasks:
        # Cancelled with work in flight. The calls are cooperative and observe
        # their own cancellation; what matters here is that whatever they
        # return still commits in order, so the history stays well-formed.
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        commit_ready()

    commit_ready()

    if batch.aborted:
        for call in group[started:]:
            _append_aborted(session, call, turn=turn, step=step, batch=batch)
        return len(group)
    return started


def _append_aborted(
    session: Session, call: ToolCall, *, turn: int, step: int, batch: ToolBatch
) -> None:
    """Record the call/result pair for a call cancellation never dispatched."""
    event = session.append(
        TOOL_CALL,
        {
            "turn": turn,
            "step": step,
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        },
    )
    result = ToolResult(
        tool=call.name,
        call_id=call.id,
        ok=False,
        content=ABORTED_TEXT,
        error=ToolError.CANCELLED,
        detail="aborted before dispatch",
    )
    log_tool_result(session, result, turn=turn, step=step, call_seq=event.seq)
    batch.results.append(result)


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Model arguments as a dict, tolerating what models actually emit.

    dsh preserves invalid JSON as text and maps empty input to `{}`. koe's
    tools take a mapping, so invalid JSON becomes `{"_raw": ...}` instead of
    being dropped: the tool then rejects it with a message the model can act on,
    which is more useful than a schema error it cannot see.
    """
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {"_raw": raw}
