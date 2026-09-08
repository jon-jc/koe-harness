"""OpenAI adapter.

Exists so the LLM layer is genuinely multi-vendor rather than nominally so: the
router can only trade one backend against another if there are two, and the
eval harness can only answer "which model writes better 議事録" if both are
reachable through the same interface.

Structured output uses `response_format` with a JSON schema, which is this
vendor's equivalent of constrained decoding. The guardrail layer downstream is
unchanged either way -- it verifies citations against the transcript and does
not care how the JSON was produced. That independence is the point of putting
verification in its own layer.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from koe.providers.base import (
    LLMResponse,
    Message,
    Modality,
    ProviderError,
    ProviderInfo,
    ProviderTimeout,
    Usage,
    measured,
)
from koe.text.script import Language

logger = logging.getLogger(__name__)

#: USD per million tokens, as (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

DEFAULT_MODEL = "gpt-4o"


def _price(model: str) -> tuple[float, float]:
    for known, rates in PRICING.items():
        if model.startswith(known):
            return rates
    logger.warning("no pricing for model %r; cost will be reported as 0", model)
    return (0.0, 0.0)


@dataclass
class OpenAILLM:
    """GPT, behind the :class:`~koe.providers.base.LLMProvider` protocol."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    max_tokens: int = 8192
    timeout_seconds: float = 120.0
    max_retries: int = 2
    name: str = "openai"
    info: ProviderInfo = field(init=False)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        input_rate, output_rate = _price(self.model)
        self.info = ProviderInfo(
            name=self.name,
            modality=Modality.LLM,
            model=self.model,
            languages=frozenset({Language.JA, Language.EN}),
            supports_streaming=True,
            cost_per_1k_input_tokens_usd=input_rate / 1000.0,
            cost_per_1k_output_tokens_usd=output_rate / 1000.0,
            typical_first_result_ms=1200.0,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        return self._client

    @staticmethod
    def is_configured() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _wrap_error(self, exc: Exception) -> ProviderError:
        import openai

        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeout(f"{self.name} timed out: {exc}", provider=self.name)
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(
                f"{self.name} rate limited: {exc}", provider=self.name, retryable=True, status=429
            )
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError(
                f"{self.name} connection error: {exc}", provider=self.name, retryable=True
            )
        if isinstance(exc, openai.APIStatusError):
            return ProviderError(
                f"{self.name} returned {exc.status_code}: {exc}",
                provider=self.name,
                retryable=exc.status_code >= 500,
                status=exc.status_code,
            )
        return ProviderError(f"{self.name} failed: {exc!r}", provider=self.name, retryable=False)

    def _usage_from(self, raw: Any) -> Usage:
        input_rate, output_rate = _price(self.model)
        prompt_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(raw, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)

        fresh = max(0, prompt_tokens - cached)
        cost = (
            fresh * input_rate + cached * input_rate * 0.5 + completion_tokens * output_rate
        ) / 1_000_000.0

        return Usage(
            provider=self.name,
            model=self.model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=cost,
            cached=cached > 0,
        )

    def _payload(self, messages: Sequence[Message], system: str | None) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend({"role": m.role, "content": m.content} for m in messages)
        return out

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 0,
        temperature: float = 0.0,
    ) -> LLMResponse:
        usage = Usage()
        try:
            async with measured(self.info, usage):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature,
                    messages=self._payload(messages, system),
                )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        latency_ms = usage.latency_ms
        usage = self._usage_from(response.usage)
        usage.latency_ms = latency_ms
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            usage=usage,
            model=response.model,
            stop_reason=choice.finish_reason or "",
            raw=response,
        )

    async def complete_json(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int = 0,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Generate JSON constrained to `schema`."""
        usage = Usage()
        try:
            async with measured(self.info, usage):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=0.0,
                    messages=self._payload(messages, system),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "minutes", "schema": schema, "strict": False},
                    },
                )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        latency_ms = usage.latency_ms
        usage = self._usage_from(response.usage)
        usage.latency_ms = latency_ms
        choice = response.choices[0]
        text = choice.message.content or ""
        reply = LLMResponse(
            text=text,
            usage=usage,
            model=response.model,
            stop_reason=choice.finish_reason or "",
            raw=response,
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.name} returned unparseable JSON: {exc}",
                provider=self.name,
                retryable=False,
            ) from exc
        return (parsed, reply)
