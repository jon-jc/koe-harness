"""A local model behind koe's LLM protocol.

Speaks OpenAI's ``/v1/chat/completions``, which Ollama, LM Studio, llama.cpp,
vLLM, Jan and LocalAI all serve. One client covers all of them.

**Cost is zero and that is not the same as free.** A local model bills nothing,
so every cost figure here is 0.0 and the ledger will show 0.0 -- correctly. What
it costs instead is latency and quality, and those are the numbers the router
has to have, or it will route every request to the local model on price and
koe will feel broken. So :attr:`typical_first_result_ms` is deliberately
pessimistic and the quality prior is deliberately worse than the hosted models',
because the router's job is to weigh those against each other and it can only do
that if local inference declares its real disadvantages.

**Reasoning output is stripped here rather than by the caller.** Local models
are disproportionately reasoning-tuned distills -- it is what fits on a laptop
-- so ``<think>`` blocks arrive far more often on this path than on a hosted
one. Stripping at the boundary means every consumer of an LLMResponse sees an
answer rather than a scratchpad.

**A 404 on the chat path is a configuration error worth naming.** The most
common local misconfiguration by a distance is a base URL pointing at the
server's origin instead of its API, because that is what the browser address
bar shows. The error says so rather than reporting "not found".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from koe.providers.base import (
    LLMResponse,
    Message,
    Modality,
    ProviderError,
    ProviderInfo,
    Usage,
    measured,
)
from koe.providers.local.endpoints import openai_base_candidates
from koe.providers.local.http import GENERATE_TIMEOUT_S, LocalHTTPError, post_json
from koe.text.script import Language
from koe.text.thinking import strip_thinking

logger = logging.getLogger(__name__)

#: What a mid-size quantized model on a consumer CPU takes to produce a first
#: token. Wildly variable -- a 3B on an M3 is far quicker, a 70B on a laptop far
#: slower -- but the router needs a number, and one that flatters local
#: inference would have it win every race it should lose.
DEFAULT_FIRST_RESULT_MS = 2_500.0

#: A prior, not a measurement. Local models are smaller than the hosted ones,
#: and Japanese is where that gap is widest: the multilingual training mix in an
#: open-weights model is mostly English, so a 7B that writes decent English
#: minutes can still produce Japanese that a reader would not accept.
DEFAULT_ERROR_RATE = {Language.EN: 0.10, Language.JA: 0.20}


@dataclass(slots=True)
class LocalLLM:
    """An OpenAI-compatible model server on this machine."""

    model: str
    base_url: str
    #: Sent when present. Local servers rarely need one; vLLM behind a proxy
    #: sometimes does.
    api_key: str = ""
    label: str = "local"
    timeout_s: float = GENERATE_TIMEOUT_S
    info: ProviderInfo = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = ProviderInfo(
                name=self.label,
                modality=Modality.LLM,
                model=self.model,
                languages=frozenset({Language.EN, Language.JA}),
                # Zero, and meant literally: nothing is billed. The router
                # weighs this against the speed and quality priors below.
                cost_per_1k_input_tokens_usd=0.0,
                cost_per_1k_output_tokens_usd=0.0,
                typical_first_result_ms=DEFAULT_FIRST_RESULT_MS,
                expected_error_rate=dict(DEFAULT_ERROR_RATE),
            )

    @property
    def endpoint(self) -> str:
        """The chat-completions URL, with the `/v1` question already settled."""
        candidates = openai_base_candidates(self.base_url)
        if not candidates:
            raise ProviderError(f"unusable base URL: {self.base_url!r}")
        return f"{candidates[-1]}/chat/completions"

    async def complete(
        self,
        messages: list[Message] | Any,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _chat_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        usage = Usage(provider=self.info.name, model=self.model)
        async with measured(self.info, usage):
            body = await self._post(payload)

        text = _first_message_text(body)
        reported = body.get("usage") if isinstance(body, dict) else None
        if isinstance(reported, dict):
            usage.input_tokens = int(reported.get("prompt_tokens") or 0)
            usage.output_tokens = int(reported.get("completion_tokens") or 0)
        usage.cost_usd = 0.0

        return LLMResponse(text=strip_thinking(text), usage=usage)

    async def _post(self, payload: dict[str, Any]) -> Any:
        try:
            return await post_json(
                self.endpoint, payload, timeout=self.timeout_s, api_key=self.api_key
            )
        except LocalHTTPError as exc:
            raise ProviderError(self._explain(exc)) from exc

    def _explain(self, exc: LocalHTTPError) -> str:
        """Turn a transport failure into something a user can act on."""
        if exc.unreachable:
            return (
                f"no local model server answered at {self.base_url} — "
                f"start it, or pick another in Settings → Models"
            )
        if exc.status == 404:
            # By far the most common local misconfiguration: the base URL is
            # the web UI's origin rather than the API's.
            return (
                f"{self.base_url} has no chat endpoint — the base URL usually needs to end in /v1"
            )
        if exc.status in (400, 422):
            return f"{self.model!r} rejected the request — is that model loaded? ({exc})"
        if exc.status in (401, 403):
            return f"{self.base_url} requires a key — add one in Settings → Models ({exc})"
        return str(exc)


def _chat_messages(messages: Any, system: str | None) -> list[dict[str, str]]:
    """koe's Message list as OpenAI chat messages.

    The system prompt goes in as a message rather than a separate field, which
    is what the OpenAI-compatible surface expects and what every local server
    implements. Servers whose template has no system role fold it into the
    first user turn themselves.
    """
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        out.append({"role": message.role, "content": message.content})
    return out


def _first_message_text(body: Any) -> str:
    """The assistant text, tolerating the shapes local servers actually emit."""
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some servers return the content-parts array from the vision API even
        # for text-only replies.
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        # llama.cpp's older completion shape.
        if isinstance(first.get("text"), str):
            return str(first["text"])
    if isinstance(first.get("text"), str):
        return str(first["text"])
    return ""
