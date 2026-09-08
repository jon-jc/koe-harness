"""Anthropic (Claude) adapter.

Three things this does beyond wrapping the SDK:

**Prompt caching on the stable prefix.** The 議事録 system prompt plus its JSON
schema runs to a few thousand tokens and is byte-identical for every meeting,
while the transcript is different every time. Marking the system block cacheable
and keeping the transcript after it means the constant part is billed at ~10% on
every call after the first. This is the single largest cost lever in the LLM
layer, and it costs one parameter.

**Errors classified by whether retrying elsewhere could help.** The router's
fallback logic depends on that distinction: a rate limit or a 5xx is worth
trying on another backend, an authentication failure or a malformed request is
not and would only burn the budget again.

**Cost computed from real usage, including cache tiers.** Cached reads and cache
writes are priced differently from fresh input tokens, so a naive
``input_tokens * rate`` overstates a cached workload by roughly an order of
magnitude and would make caching look useless in the cost report.
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

#: Per-model pricing in USD per million tokens, as (input, output).
#: Kept explicit rather than fetched so a cost report is reproducible offline.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Cache reads bill at ~10% of the input rate; cache writes at ~125%.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

DEFAULT_MODEL = "claude-opus-5"


def _price(model: str) -> tuple[float, float]:
    for known, rates in PRICING.items():
        if model.startswith(known):
            return rates
    logger.warning("no pricing for model %r; cost will be reported as 0", model)
    return (0.0, 0.0)


@dataclass
class AnthropicLLM:
    """Claude, behind the :class:`~koe.providers.base.LLMProvider` protocol."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    max_tokens: int = 8192
    effort: str = "high"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    #: Cache the system prompt. Worth it whenever the system block is stable
    #: and non-trivial, which is the case for every prompt in this codebase.
    cache_system: bool = True
    name: str = "anthropic"
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
            typical_first_result_ms=1500.0,
        )

    # -- client --------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Lazily constructed SDK client.

        Lazy so that importing koe -- which the CLI and the eval harness do on
        every invocation -- never requires credentials to be present.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        return self._client

    @staticmethod
    def is_configured() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # -- error mapping -------------------------------------------------------

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Classify an SDK exception by whether another backend could succeed."""
        import anthropic

        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderTimeout(f"{self.name} timed out: {exc}", provider=self.name)
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(
                f"{self.name} rate limited: {exc}",
                provider=self.name,
                retryable=True,
                status=429,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(
                f"{self.name} connection error: {exc}", provider=self.name, retryable=True
            )
        if isinstance(exc, anthropic.APIStatusError):
            # 5xx is the provider's problem and may succeed elsewhere; 4xx is
            # this request's problem and will fail identically everywhere.
            retryable = exc.status_code >= 500
            return ProviderError(
                f"{self.name} returned {exc.status_code}: {exc}",
                provider=self.name,
                retryable=retryable,
                status=exc.status_code,
            )
        return ProviderError(f"{self.name} failed: {exc!r}", provider=self.name, retryable=False)

    # -- usage / cost --------------------------------------------------------

    def _usage_from(self, raw: Any) -> Usage:
        input_rate, output_rate = _price(self.model)
        fresh_input = int(getattr(raw, "input_tokens", 0) or 0)
        cache_read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
        output = int(getattr(raw, "output_tokens", 0) or 0)

        cost = (
            fresh_input * input_rate
            + cache_read * input_rate * CACHE_READ_MULTIPLIER
            + cache_write * input_rate * CACHE_WRITE_MULTIPLIER
            + output * output_rate
        ) / 1_000_000.0

        return Usage(
            provider=self.name,
            model=self.model,
            input_tokens=fresh_input + cache_read + cache_write,
            output_tokens=output,
            cost_usd=cost,
            cached=cache_read > 0,
        )

    # -- requests ------------------------------------------------------------

    def _system_blocks(self, system: str | None) -> Any:
        if not system:
            return None
        if not self.cache_system:
            return system
        # A cacheable prefix: identical across every meeting, so after the first
        # call the schema and instructions bill at ~10%.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 0,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion."""
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            # Adaptive thinking: the model decides how much reasoning a given
            # transcript needs. A six-line standup and a contentious hour-long
            # planning meeting do not deserve the same budget.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        blocks = self._system_blocks(system)
        if blocks is not None:
            payload["system"] = blocks

        usage = Usage()
        try:
            async with measured(self.info, usage):
                response = await self.client.messages.create(**payload)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        latency_ms = usage.latency_ms
        usage = self._usage_from(response.usage)
        usage.latency_ms = latency_ms

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            usage=usage,
            model=getattr(response, "model", self.model),
            stop_reason=getattr(response, "stop_reason", "") or "",
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
        """Generate a completion constrained to `schema`.

        Uses the API's structured-output support so the response is
        schema-valid at the source, rather than parsed hopefully at this end.
        The guardrail layer still validates -- constrained decoding guarantees
        the *shape*, never that the content is grounded in the transcript.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        blocks = self._system_blocks(system)
        if blocks is not None:
            payload["system"] = blocks

        usage = Usage()
        try:
            async with measured(self.info, usage):
                response = await self.client.messages.create(**payload)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        latency_ms = usage.latency_ms
        usage = self._usage_from(response.usage)
        usage.latency_ms = latency_ms

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        reply = LLMResponse(
            text=text,
            usage=usage,
            model=getattr(response, "model", self.model),
            stop_reason=getattr(response, "stop_reason", "") or "",
            raw=response,
        )

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # Non-retryable: the same prompt will produce the same malformed
            # output at another provider. The generator repairs instead.
            raise ProviderError(
                f"{self.name} returned unparseable JSON: {exc}",
                provider=self.name,
                retryable=False,
            ) from exc
        return (parsed, reply)

    async def count_tokens(self, messages: Sequence[Message], *, system: str | None = None) -> int:
        """Count input tokens without generating.

        Used to decide whether a long meeting needs chunking, before paying to
        find out that it did.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            payload["system"] = system
        try:
            result = await self.client.messages.count_tokens(**payload)
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return int(result.input_tokens)
