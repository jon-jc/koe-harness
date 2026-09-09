"""The agent loop.

The turn/step boundary is what these tests defend. Treating one model request
as the unit means tool results are never seen by the model that asked for
them; letting the loop run unbounded means a model calling a failing tool
forever spends the budget. Both failures look like "it works" in a happy-path
test, so both get one of their own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from koe.agent import Adapter, ChatMessage, Conversation, ModelReply, ToolCall
from koe.agent.adapters import (
    MockAdapter,
    _anthropic_messages,
    _openai_messages,
    _search_term,
    adapter_for,
)
from koe.tools import ToolRegistry, ToolRun, ToolSpec

pytestmark = pytest.mark.asyncio


class Scripted(Adapter):
    """An adapter that replays a fixed list of replies."""

    name = "scripted"

    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[ChatMessage]] = []

    async def reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
    ) -> ModelReply:
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else ModelReply(text="done")


def registry_with(name: str = "probe", body: Any = None) -> ToolRegistry:
    async def default(args: dict[str, Any], run: ToolRun) -> Any:
        return f"ran {name} with {args}"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            description="A probe.",
            parameters={"type": "object", "properties": {}},
            execute=body or default,
        )
    )
    return registry


# --------------------------------------------------------------------------
# the turn boundary
# --------------------------------------------------------------------------


async def test_a_turn_with_no_tools_is_one_step() -> None:
    conversation = Conversation(Scripted([ModelReply(text="hello")]), ToolRegistry())
    result = await conversation.send("hi")

    assert result.text == "hello"
    assert result.steps == 1
    assert result.tool_calls == 0


async def test_a_tool_call_costs_another_step() -> None:
    """The point of a turn: the model must see what its tool returned."""
    adapter = Scripted(
        [
            ModelReply(tool_calls=[ToolCall(id="1", name="probe", arguments={})]),
            ModelReply(text="the probe said something"),
        ]
    )
    conversation = Conversation(adapter, registry_with())

    result = await conversation.send("use the probe")

    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.text == "the probe said something"


async def test_the_model_actually_receives_the_tool_result() -> None:
    """The failure this guards is silent: the model answers without the data."""
    adapter = Scripted(
        [
            ModelReply(tool_calls=[ToolCall(id="1", name="probe", arguments={"x": 1})]),
            ModelReply(text="done"),
        ]
    )
    conversation = Conversation(adapter, registry_with())
    await conversation.send("go")

    second_request = adapter.seen[1]
    assert second_request[-1].role == "tool"
    assert "ran probe" in second_request[-1].tool_results[0].content


async def test_several_calls_in_one_step_come_back_together() -> None:
    """A vendor rejects a request that answers only some of its calls."""
    adapter = Scripted(
        [
            ModelReply(
                tool_calls=[
                    ToolCall(id="1", name="probe", arguments={}),
                    ToolCall(id="2", name="probe", arguments={}),
                ]
            ),
            ModelReply(text="done"),
        ]
    )
    conversation = Conversation(adapter, registry_with())

    result = await conversation.send("go")

    assert result.tool_calls == 2
    assert len(adapter.seen[1][-1].tool_results) == 2


async def test_a_runaway_model_is_stopped_and_says_so() -> None:
    """Unbounded, a model that keeps calling a tool spends until the budget is gone."""
    always = [ModelReply(tool_calls=[ToolCall(id="x", name="probe", arguments={})])] * 20
    conversation = Conversation(Scripted(always), registry_with(), max_steps=4)

    result = await conversation.send("loop forever")

    assert result.steps == 4
    assert result.stopped_at_limit
    assert "4 steps" in result.text


# --------------------------------------------------------------------------
# failure
# --------------------------------------------------------------------------


async def test_a_failing_tool_does_not_end_the_turn() -> None:
    """A person would try something else; so should the loop."""

    async def explode(args: dict[str, Any], run: ToolRun) -> Any:
        raise RuntimeError("nope")

    adapter = Scripted(
        [
            ModelReply(tool_calls=[ToolCall(id="1", name="probe", arguments={})]),
            ModelReply(text="I recovered"),
        ]
    )
    conversation = Conversation(adapter, registry_with(body=explode))

    result = await conversation.send("go")

    assert result.text == "I recovered"
    assert not adapter.seen[1][-1].tool_results[0].ok


async def test_a_model_error_is_reported_rather_than_raised() -> None:
    class Broken(Adapter):
        name = "broken"

        async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
            raise ConnectionError("the vendor is down")

    conversation = Conversation(Broken(), ToolRegistry())
    result = await conversation.send("hi")

    assert "the vendor is down" in result.text


async def test_an_unknown_tool_comes_back_as_a_result() -> None:
    adapter = Scripted(
        [
            ModelReply(tool_calls=[ToolCall(id="1", name="imaginary", arguments={})]),
            ModelReply(text="ok"),
        ]
    )
    conversation = Conversation(adapter, registry_with())

    await conversation.send("go")

    assert adapter.seen[1][-1].tool_results[0].error is not None


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


async def test_the_turn_emits_as_it_goes() -> None:
    """A loop that only returns at the end can only be watched by waiting."""
    adapter = Scripted(
        [
            ModelReply(tool_calls=[ToolCall(id="1", name="probe", arguments={})]),
            ModelReply(text="done"),
        ]
    )
    conversation = Conversation(adapter, registry_with())
    events: list[str] = []

    async def listen(name: str, payload: dict[str, Any]) -> None:
        events.append(name)

    await conversation.send("go", on_event=listen)

    assert events == [
        "turn/start",
        "step/start",
        "tool/call",
        "tool/result",
        "step/start",
        "assistant/text",
        "turn/end",
    ]


async def test_history_survives_between_turns() -> None:
    conversation = Conversation(
        Scripted([ModelReply(text="a"), ModelReply(text="b")]), ToolRegistry()
    )
    await conversation.send("first")
    await conversation.send("second")

    roles = [m["role"] for m in conversation.history()]
    assert roles == ["user", "assistant", "user", "assistant"]


# --------------------------------------------------------------------------
# the mock adapter
# --------------------------------------------------------------------------


async def test_the_mock_really_dispatches_a_tool() -> None:
    """A mock that only returned prose would let the tool path rot untested."""
    conversation = Conversation(MockAdapter(), registry_with("list_files"))
    result = await conversation.send("what files are here?")

    assert result.tool_calls == 1
    assert "ran list_files" in result.text


async def test_the_mock_says_what_it_is() -> None:
    """Nobody should mistake the deterministic answer for a model's."""
    conversation = Conversation(MockAdapter(), ToolRegistry())
    result = await conversation.send("hello")

    assert "mock" in result.text.lower()
    assert "API key" in result.text


async def test_the_mock_reports_a_tool_failure_rather_than_inventing_an_answer() -> None:
    async def explode(args: dict[str, Any], run: ToolRun) -> Any:
        raise RuntimeError("disk gone")

    conversation = Conversation(MockAdapter(), registry_with("list_files", body=explode))
    result = await conversation.send("what files are here?")

    assert "failed" in result.text


def test_the_search_term_heuristic_prefers_quoted_text() -> None:
    assert _search_term('search for "retry_after" please') == "retry_after"
    assert _search_term("find 「議事録」 in the code") == "議事録"


def test_the_search_term_falls_back_to_the_longest_word() -> None:
    assert _search_term("grep for tokenization") == "tokenization"


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


def test_an_unknown_provider_gets_the_mock() -> None:
    """Better a deterministic assistant than a crash on an unrecognised name."""
    assert adapter_for(object()).name == "mock"


def test_openai_results_become_one_message_each() -> None:
    """OpenAI wants a `tool` message per result; Anthropic wants one block list."""
    from koe.tools.registry import ToolResult

    messages = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="a", name="probe", arguments={"k": "v"})],
        ),
        ChatMessage(
            role="tool",
            tool_results=[
                ToolResult(tool="probe", call_id="a", ok=True, content="out"),
                ToolResult(tool="probe", call_id="b", ok=True, content="out2"),
            ],
        ),
    ]

    rendered = _openai_messages(messages)

    assert [m["role"] for m in rendered] == ["user", "assistant", "tool", "tool"]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == '{"k": "v"}'


def test_anthropic_results_become_one_user_message_of_blocks() -> None:
    """The detail most easily got wrong when porting from the OpenAI shape."""
    from koe.tools.registry import ToolResult

    messages = [
        ChatMessage(role="user", content="go"),
        ChatMessage(role="assistant", tool_calls=[ToolCall(id="a", name="probe", arguments={})]),
        ChatMessage(
            role="tool",
            tool_results=[ToolResult(tool="probe", call_id="a", ok=True, content="out")],
        ),
    ]

    rendered = _anthropic_messages(messages)

    assert [m["role"] for m in rendered] == ["user", "assistant", "user"]
    assert rendered[2]["content"][0]["type"] == "tool_result"
    assert rendered[2]["content"][0]["tool_use_id"] == "a"


def test_an_empty_assistant_turn_is_not_sent_to_anthropic() -> None:
    """A message with neither text nor calls fails the whole request."""
    rendered = _anthropic_messages([ChatMessage(role="assistant", content="")])
    assert rendered == []


def test_a_tool_error_is_flagged_for_anthropic() -> None:
    from koe.tools.registry import ToolError, ToolResult

    rendered = _anthropic_messages(
        [
            ChatMessage(
                role="tool",
                tool_results=[
                    ToolResult(
                        tool="probe",
                        call_id="a",
                        ok=False,
                        content="boom",
                        error=ToolError.FAILED,
                    )
                ],
            )
        ]
    )

    assert rendered[0]["content"][0]["is_error"] is True
