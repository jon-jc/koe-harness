"""The harness: session log, inbox, scheduling, and the turn/step machine.

Ported behaviour from DeepSeek Harness, so the tests are written against the
properties that port exists to provide rather than against the implementation
that provides them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from koe.agent.loop import Adapter, ChatMessage, ModelReply, ToolCall
from koe.harness import HarnessAgent, Inbox, Session
from koe.harness.inbox import NEXT_STEP, NEXT_TURN
from koe.harness.session import ASSISTANT_MESSAGE, TOOL_CALL, TOOL_RESULT, TURN_START, USER_MESSAGE
from koe.kernel.context import Context
from koe.tools.registry import ToolRegistry, ToolSpec


class Scripted(Adapter):
    """Replies from a script, so the turn structure is what is under test."""

    name = "scripted"

    def __init__(self, *script: ModelReply) -> None:
        self.script = list(script)
        self.requests: list[list[ChatMessage]] = []
        self.systems: list[str] = []

    async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
        self.requests.append(list(messages))
        self.systems.append(system)
        return self.script.pop(0) if self.script else ModelReply(text="done")


def tool_registry(order: list[str] | None = None, *, delay: float = 0.0) -> ToolRegistry:
    log = order if order is not None else []
    tools = ToolRegistry(Context(name="t"))

    async def read(args: dict[str, Any], run: Any) -> str:
        path = str(args.get("path", ""))
        # Inverted delays: the later call finishes first, so commit ordering is
        # observable rather than accidental.
        await asyncio.sleep(delay * (2 if path == "a" else 1))
        log.append(f"read:{path}")
        return f"contents of {path}"

    async def shell(args: dict[str, Any], run: Any) -> str:
        log.append("shell:start")
        await asyncio.sleep(delay)
        log.append("shell:end")
        return "ran"

    tools.register(
        ToolSpec(
            name="read",
            description="read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            execute=read,
            parallel_safe=True,
        )
    )
    tools.register(
        ToolSpec(
            name="shell",
            description="run a command",
            parameters={"type": "object", "properties": {}},
            execute=shell,
            parallel_safe=False,
        )
    )
    return tools


def calls(*specs: tuple[str, str, dict[str, Any]]) -> ModelReply:
    return ModelReply(
        text="", tool_calls=[ToolCall(id=i, name=n, arguments=a) for i, n, a in specs]
    )


# --------------------------------------------------------------------------
# the session log
# --------------------------------------------------------------------------


def test_the_history_is_derived_from_the_log_not_stored() -> None:
    """The central claim of the port. Everything else depends on it."""
    session = Session(system="be helpful")
    session.append(USER_MESSAGE, {"text": "hello"})
    session.append(ASSISTANT_MESSAGE, {"text": "hi"})

    messages = session.derive_messages()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert session.system_prompt() == "be helpful"


def test_a_tool_call_and_its_result_rejoin_as_the_pair_a_vendor_expects() -> None:
    """They are separate facts in the log -- a call is a fact the moment the
    model asks, answered or not -- and only the fold gathers them back."""
    session = Session()
    session.append(USER_MESSAGE, {"text": "read it"})
    # The anchor is written before the calls it made: the step commits the
    # assistant text first, so a cancellation cannot lose it.
    session.append(ASSISTANT_MESSAGE, {"text": ""})
    session.append(TOOL_CALL, {"call_id": "1", "name": "read", "arguments": {}})
    session.append(TOOL_RESULT, {"call_id": "1", "name": "read", "ok": True, "content": "x"})

    messages = session.derive_messages()
    assert [m.role for m in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls[0].name == "read"
    assert messages[2].tool_results[0].content == "x"


def test_the_prompt_is_replaced_by_appending_not_by_editing() -> None:
    """A correction is a later fact that shadows an earlier one; nothing in the
    log is ever retracted."""
    session = Session(system="first")
    session.append("system/message", {"text": "second"})
    assert session.system_prompt() == "second"
    assert len(session.of_type("system/message")) == 2


def test_a_result_cites_the_call_it_answers() -> None:
    """So a UI pairs them without matching on an id the model chose."""
    session = Session()
    call = session.append(TOOL_CALL, {"call_id": "1", "name": "read", "arguments": {}})
    result = session.append(TOOL_RESULT, {"call_id": "1"}, sources=[call.seq])
    assert result.sources == (call.seq,)


def test_a_closed_session_refuses_writes_but_still_reads() -> None:
    session = Session()
    session.append(USER_MESSAGE, {"text": "hi"})
    session.close()
    with pytest.raises(RuntimeError):
        session.append(USER_MESSAGE, {"text": "again"})
    assert len(session.derive_messages()) == 1


# --------------------------------------------------------------------------
# the inbox
# --------------------------------------------------------------------------


def test_the_queue_is_a_fold_over_the_log() -> None:
    """Not stored twice. The log is the queue."""
    session = Session()
    inbox = Inbox(session)
    inbox.followup("one")
    inbox.followup("two")

    assert [m.text for m in inbox.next_turn] == ["one", "two"]
    # Rebuilt from scratch: same answer.
    assert [m.text for m in Inbox(session).next_turn] == ["one", "two"]


def test_a_turn_boundary_takes_one_prompt_and_all_steering() -> None:
    """One queue is drained and the other is sipped: taking two prompts at a
    turn boundary would merge two things the user asked separately."""
    session = Session()
    inbox = Inbox(session)
    inbox.followup("first")
    inbox.followup("second")
    inbox.steer("while you work")

    claimed = inbox.claim(NEXT_TURN, turn=1)

    assert [m.text for m in claimed] == ["while you work", "first"]
    assert [m.text for m in inbox.next_turn] == ["second"]


def test_a_step_boundary_takes_only_steering() -> None:
    session = Session()
    inbox = Inbox(session)
    inbox.followup("next turn's work")
    inbox.steer("this turn's correction")

    claimed = inbox.claim(NEXT_STEP, turn=1)

    assert [m.text for m in claimed] == ["this turn's correction"]
    assert [m.text for m in inbox.next_turn] == ["next turn's work"]


def test_injected_context_is_not_the_user_speaking() -> None:
    """It reads identically to the model and must not read identically in the
    UI, or the transcript shows the user saying things they never said."""
    inbox = Inbox(Session())
    inbox.steer("from a person")
    inbox.inject("from a tool")
    assert [m.kind for m in inbox.next_step] == ["user", "context"]


def test_a_malformed_splice_is_refused_rather_than_guessed() -> None:
    """Guessing would put the agent to work on invented input."""
    session = Session()
    inbox = Inbox(session)
    session.append("agent/inbox/spliced", {"target": NEXT_TURN, "start": 9, "removed": 0})
    with pytest.raises(ValueError, match="invalid inbox splice"):
        _ = inbox.next_turn


# --------------------------------------------------------------------------
# tool scheduling
# --------------------------------------------------------------------------


async def test_parallel_safe_calls_overlap() -> None:
    order: list[str] = []
    agent = HarnessAgent(
        Scripted(calls(("1", "read", {"path": "a"}), ("2", "read", {"path": "b"}))),
        tool_registry(order, delay=0.04),
    )
    started = asyncio.get_running_loop().time()
    await agent.ask("read both")
    elapsed = asyncio.get_running_loop().time() - started

    # Two 40-80ms calls overlapping take about the longer one, not the sum.
    assert elapsed < 0.16, f"{elapsed:.3f}s suggests serial execution"
    assert order == ["read:b", "read:a"], "the later call should finish first"


async def test_results_commit_in_model_order_whatever_the_finish_order() -> None:
    """The rule that makes parallelism invisible to the model. Appending as
    they land would make the derived history depend on disk timing, so the same
    conversation replays differently and a prefix cache misses."""
    order: list[str] = []
    agent = HarnessAgent(
        Scripted(calls(("1", "read", {"path": "a"}), ("2", "read", {"path": "b"}))),
        tool_registry(order, delay=0.03),
    )
    await agent.ask("read both")

    assert order == ["read:b", "read:a"], "precondition: finishes out of order"
    logged = [e.data["call_id"] for e in agent.session.of_type(TOOL_RESULT)]
    assert logged == ["1", "2"], "committed out of model order"


async def test_an_exclusive_tool_is_a_barrier() -> None:
    """A shell command can change the working directory or take a lock, so it
    runs alone -- and the calls on either side do not cross it."""
    order: list[str] = []
    agent = HarnessAgent(
        Scripted(
            calls(
                ("1", "read", {"path": "a"}),
                ("2", "shell", {}),
                ("3", "read", {"path": "b"}),
            )
        ),
        tool_registry(order, delay=0.02),
    )
    await agent.ask("read, run, read")

    assert order == ["read:a", "shell:start", "shell:end", "read:b"]


async def test_every_call_gets_a_result_even_when_cancelled() -> None:
    """An assistant turn containing a call with no result is a request vendors
    reject, so the next turn would fail on the history rather than on the
    cancellation."""
    agent = HarnessAgent(
        Scripted(
            calls(
                ("1", "shell", {}),
                ("2", "shell", {}),
                ("3", "shell", {}),
            )
        ),
        tool_registry(delay=0.05),
    )
    agent.followup("go")
    await asyncio.sleep(0.02)
    agent.cancel("user")
    await agent.wait_idle()

    logged_calls = agent.session.of_type(TOOL_CALL)
    logged_results = agent.session.of_type(TOOL_RESULT)
    assert len(logged_calls) == len(logged_results) == 3
    aborted = [e for e in logged_results if e.data.get("error") == "cancelled"]
    assert aborted, "calls that never dispatched should carry a synthetic result"


# --------------------------------------------------------------------------
# the turn/step machine
# --------------------------------------------------------------------------


async def test_a_turn_runs_steps_until_the_model_owes_nothing() -> None:
    agent = HarnessAgent(
        Scripted(calls(("1", "read", {"path": "a"})), ModelReply(text="the answer")),
        tool_registry(),
    )
    outcome = await agent.ask("go")

    assert outcome.reason == "completed"
    assert outcome.steps == 2
    assert outcome.text == "the answer"


async def test_steering_joins_the_running_turn_rather_than_starting_one() -> None:
    """The point of the inbox. A model three calls into the wrong file needs
    telling while it works, not cancelling and re-prompting."""
    order: list[str] = []
    agent = HarnessAgent(
        Scripted(calls(("1", "read", {"path": "a"})), ModelReply(text="corrected")),
        tool_registry(order, delay=0.04),
    )
    agent.followup("look at a")
    await asyncio.sleep(0.03)
    agent.steer("actually look at b")
    await agent.wait_idle()

    assert len(agent.session.of_type(TURN_START)) == 1, "steering started a second turn"
    said = [e.data["text"] for e in agent.session.of_type(USER_MESSAGE)]
    assert said == ["look at a", "actually look at b"]


async def test_what_the_user_saw_is_kept_when_a_turn_is_cancelled() -> None:
    """Discarding it would leave the model and the user with different
    histories, and the model would be the one that is wrong."""
    agent = HarnessAgent(
        Scripted(ModelReply(text="partial answer", tool_calls=[ToolCall("1", "shell", {})])),
        tool_registry(delay=0.05),
    )
    agent.followup("go")
    await asyncio.sleep(0.02)
    agent.cancel("user")
    await agent.wait_idle()

    assistant = agent.session.of_type(ASSISTANT_MESSAGE)[-1]
    assert assistant.data["text"] == "partial answer"
    assert agent.session.of_type("turn/end")[-1].data["reason"] == "cancelled"


async def test_cancelling_one_answer_keeps_what_was_queued_behind_it() -> None:
    agent = HarnessAgent(Scripted(ModelReply(text="a")), tool_registry())
    agent.followup("first")
    agent.followup("second")
    agent.cancel("user", keep_inbox=True)
    assert agent.inbox.pending


async def test_cancelling_entirely_clears_the_queue() -> None:
    agent = HarnessAgent(Scripted(ModelReply(text="a")), tool_registry())
    agent.followup("first")
    agent.followup("second")
    agent.cancel("user")
    assert not agent.inbox.pending


async def test_a_runaway_turn_stops_at_the_step_bound() -> None:
    """A bound, not a target: only a model that keeps calling tools reaches
    it, and reaching it is a reported outcome rather than a silent stop."""

    class Insatiable(Adapter):
        name = "insatiable"

        async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
            return calls(("1", "read", {"path": "a"}))

    agent = HarnessAgent(Insatiable(), tool_registry(), max_steps=3)
    outcome = await agent.ask("go")

    assert outcome.reason == "max-steps"
    assert outcome.steps == 3


async def test_a_failing_provider_ends_the_turn_not_the_agent() -> None:
    """A chat surface that dies on one bad request is one nobody trusts with
    the next one."""

    class Broken(Adapter):
        name = "broken"
        calls_made = 0

        async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
            Broken.calls_made += 1
            if Broken.calls_made == 1:
                raise RuntimeError("provider exploded")
            return ModelReply(text="recovered")

    agent = HarnessAgent(Broken(), tool_registry())
    first = await agent.ask("go")
    assert first.reason == "error"

    second = await agent.ask("again")
    assert second.reason == "completed"
    assert second.text == "recovered"


async def test_the_request_header_is_logged_only_when_it_changes() -> None:
    """A twelve-step turn writes one header, not twelve -- and what varies
    between steps is config and schemas, never the prompt."""
    agent = HarnessAgent(
        Scripted(
            calls(("1", "read", {"path": "a"})),
            calls(("2", "read", {"path": "b"})),
            ModelReply(text="done"),
        ),
        tool_registry(),
    )
    await agent.ask("go")
    assert len(agent.session.of_type("request/header")) == 1


async def test_the_model_sees_the_derived_history_on_the_next_step() -> None:
    adapter = Scripted(calls(("1", "read", {"path": "a"})), ModelReply(text="done"))
    agent = HarnessAgent(adapter, tool_registry(), system="be koe")
    await agent.ask("go")

    second = adapter.requests[1]
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert adapter.systems[1] == "be koe"


async def test_an_agent_woken_with_an_empty_queue_spends_no_model_call() -> None:
    """The queue was cleared between the wake and the claim. The turn owns the
    boundary and costs nothing."""
    adapter = Scripted(ModelReply(text="should not run"))
    agent = HarnessAgent(adapter, tool_registry())
    agent.followup("go")
    agent.inbox.clear()
    await agent.wait_idle()

    assert adapter.requests == []


# --------------------------------------------------------------------------
# the context gauge
# --------------------------------------------------------------------------


async def test_every_turn_end_reports_how_full_the_window_is() -> None:
    """So a client can show context pressure continuously, rather than only
    when someone thinks to ask with /context."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def record(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    agent = HarnessAgent(Scripted(ModelReply(text="hello")), tool_registry(), on_event=record)
    await agent.ask("go")

    ends = [payload for event, payload in events if event == "turn/end"]
    assert len(ends) == 1
    context = ends[0]["context"]
    assert context["total"] > 0
    assert context["total"] == context["message_tokens"] + context["schema_tokens"]
    assert 0.0 <= context["pressure"] <= 1.0


async def test_a_broken_gauge_does_not_end_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gauge is for watching. A turn that failed because its meter did
    would be a worse bug than a missing number."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def record(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    agent = HarnessAgent(Scripted(ModelReply(text="fine")), tool_registry(), on_event=record)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("meter broke")

    monkeypatch.setattr(type(agent.compactor.meter), "measure", broken)
    outcome = await agent.ask("go")

    assert outcome.reason == "completed"
    assert outcome.text == "fine"
    ends = [payload for event, payload in events if event == "turn/end"]
    assert ends[-1]["context"] is None


async def test_a_result_frame_names_the_call_frame_it_answers() -> None:
    """A client pairs them by id. When the two frames disagreed about the key,
    every tool row in the chat sat on "running" forever while the answer
    arrived underneath it."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def record(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    agent = HarnessAgent(
        Scripted(calls(("c1", "read", {"path": "a"})), ModelReply(text="done")),
        tool_registry(),
        on_event=record,
    )
    await agent.ask("go")

    call = next(payload for event, payload in events if event == "tool/call")
    result = next(payload for event, payload in events if event == "tool/result")
    assert result["id"] == call["id"] == "c1"
    assert call["arguments"] == {"path": "a"}
    assert result["ok"] is True
    assert result["duration_ms"] >= 0
