"""Vendor adapters: three ways to ask a model for a reply.

Anthropic and OpenAI disagree about nearly everything in tool calling — the
message shape, where a result goes, what a stop reason is called, whether a
tool's arguments arrive parsed or as a JSON string. Each adapter absorbs one
vendor's opinions and hands the loop the same :class:`ModelReply`, so the loop
never learns which one it is talking to.

The mock adapter is the interesting one. It is not a stub that returns fixed
text: it **actually dispatches tools**, using keyword rules over the message,
and produces its answer from what those tools returned. That is deliberate. A
mock that only returns prose would let the tool-calling path rot untested and
would leave the chat surface undemonstrable without a paid key — and "you need
an API key before you can see whether this works" is the wrong first
experience for anyone opening the repository.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from koe.agent.loop import Adapter, ChatMessage, ModelReply, ToolCall, render_tool_results

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicAdapter(Adapter):
    """Claude, with tool use."""

    name = "anthropic"

    def __init__(self, provider: Any, *, max_tokens: int = 4096) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    async def reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
    ) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self._provider.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": _anthropic_messages(messages),
        }
        if tools:
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]

        response = await self._provider.client.messages.create(**payload)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", "")
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        usage = getattr(response, "usage", None)
        return ModelReply(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=getattr(response, "stop_reason", "") or "",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


def _anthropic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Render history in Anthropic's content-block shape.

    Tool results are `user` messages containing `tool_result` blocks — not a
    role of their own, which is the detail most easily got wrong when porting
    from the OpenAI shape.
    """
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            rendered.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            # An assistant turn with neither text nor calls is not a message
            # the vendor accepts, and sending one fails the whole request.
            if blocks:
                rendered.append({"role": "assistant", "content": blocks})
        elif message.role == "tool":
            rendered.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content or "(no output)",
                            "is_error": not result.ok,
                        }
                        for result in message.tool_results
                    ],
                }
            )
    return rendered


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class OpenAIAdapter(Adapter):
    """GPT, with function calling."""

    name = "openai"

    def __init__(self, provider: Any, *, max_tokens: int = 4096) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    async def reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
    ) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self._provider.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "system", "content": system}, *_openai_messages(messages)],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]

        response = await self._provider.client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            # Arguments arrive as a JSON *string* here, where Anthropic sends
            # them parsed. A model occasionally emits invalid JSON, and an
            # empty dict lets the tool reject it with a message the model can
            # act on rather than crashing the turn on a decode error.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (TypeError, ValueError):
                logger.warning("model produced invalid tool arguments for %s", call.function.name)
                arguments = {}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

        usage = getattr(response, "usage", None)
        return ModelReply(
            text=(message.content or "").strip(),
            tool_calls=calls,
            stop_reason=getattr(choice, "finish_reason", "") or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def _openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Render history in OpenAI's shape: a `tool` role, one message per result."""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            rendered.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            rendered.append(entry)
        elif message.role == "tool":
            for result in message.tool_results:
                rendered.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content or "(no output)",
                    }
                )
    return rendered


# --------------------------------------------------------------------------
# the mock
# --------------------------------------------------------------------------

#: Keyword rules mapping an intent to a tool call. Ordered: the first match
#: wins, so the more specific patterns come first.
_RULES: list[tuple[re.Pattern[str], str, dict[str, Any]]] = [
    (re.compile(r"議事録|minutes|decision|action item", re.I), "current_minutes", {}),
    (re.compile(r"transcript|文字起こし|said|発言", re.I), "current_transcript", {}),
    (re.compile(r"\brun\b|command|shell|ターミナル|実行", re.I), "terminal_list", {}),
    (re.compile(r"search|grep|find .*(in|inside)|検索", re.I), "grep_files", {}),
    (re.compile(r"\bfiles?\b|directory|folder|ファイル|一覧", re.I), "list_files", {}),
]


class MockAdapter(Adapter):
    """A deterministic assistant that really calls tools.

    Two steps, mirroring the shape of a real turn: pick a tool from the
    request, then answer from what it returned. The point is that the loop,
    the registry, the pipeline and the UI are all exercised on a clean
    checkout with no key — the same reason the ASR and 議事録 paths have mocks.
    """

    name = "mock"

    async def reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str,
        tools: Sequence[dict[str, Any]],
    ) -> ModelReply:
        available = {tool["name"] for tool in tools}

        # Second step: results are in hand, so answer from them.
        if messages and messages[-1].role == "tool":
            results = messages[-1].tool_results
            body = render_tool_results(results)
            failed = [r for r in results if not r.ok]
            if failed:
                return ModelReply(
                    text=(
                        f"I tried `{failed[0].tool}` and it failed: {failed[0].detail}\n\n"
                        "(This is the deterministic mock assistant — add an API key in "
                        "Settings → Models for a real one.)"
                    ),
                    input_tokens=len(body) // 4,
                    output_tokens=32,
                )
            return ModelReply(
                text=(
                    f"Here is what `{results[0].tool}` returned:\n\n{results[0].content}\n\n"
                    "(This is the deterministic mock assistant — add an API key in "
                    "Settings → Models for a real one.)"
                ),
                input_tokens=len(body) // 4,
                output_tokens=48,
            )

        question = messages[-1].content if messages else ""
        for pattern, tool, arguments in _RULES:
            if tool in available and pattern.search(question):
                if tool == "grep_files":
                    arguments = {"pattern": _search_term(question)}
                return ModelReply(
                    text="",
                    tool_calls=[ToolCall(id=f"mock_{tool}", name=tool, arguments=arguments)],
                    input_tokens=len(question) // 4,
                    output_tokens=16,
                )

        listed = ", ".join(sorted(available)) or "none"
        return ModelReply(
            text=(
                "I am the deterministic mock assistant, so I answer from tools rather "
                "than from a model. Ask me about this session's transcript or 議事録, "
                "or about the files in the workspace.\n\n"
                f"Tools available right now: {listed}.\n\n"
                "Add an API key in Settings → Models for a real assistant."
            ),
            input_tokens=len(question) // 4,
            output_tokens=64,
        )


def _search_term(question: str) -> str:
    """The most plausible search term in a question.

    Quoted text wins, then the longest word — a crude heuristic that is honest
    about being one, and only ever feeds a mock.
    """
    quoted = re.search(r"[\"'『「]([^\"'』」]{2,})[\"'』」]", question)
    if quoted:
        return quoted.group(1)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", question)
    return max(words, key=len) if words else "koe"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def adapter_for(provider: Any) -> Adapter:
    """The adapter matching whichever LLM is configured.

    Dispatches on the provider's own `name` rather than on its class, so a
    replacement that speaks the same vendor API is picked up without editing
    this function.
    """
    name = getattr(provider, "name", "") or getattr(getattr(provider, "info", None), "name", "")
    if name == "anthropic":
        return AnthropicAdapter(provider)
    if name == "openai":
        return OpenAIAdapter(provider)
    return MockAdapter()
