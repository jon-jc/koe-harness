"""Adapter rendering and the mock's heuristics.

Pure functions, so they live apart from the loop tests: those carry a
module-level asyncio mark, and a synchronous test under one makes pytest warn
on every single one.

The two vendor renderers get equal billing here because they disagree, and
the Anthropic shape is what you get wrong when porting from the OpenAI one.
"""

from __future__ import annotations

from koe.agent import ChatMessage, ToolCall
from koe.agent.adapters import (
    _anthropic_messages,
    _openai_messages,
    _search_term,
    adapter_for,
)
from koe.tools.registry import ToolError, ToolResult


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
