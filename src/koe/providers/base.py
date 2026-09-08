"""Provider protocols and the metadata the router selects on.

koe treats an ASR backend, a diarizer and an LLM as interchangeable
implementations behind narrow protocols, so that "which model runs this
request" stays a runtime decision.

The interesting part is :class:`ProviderInfo`. A router cannot choose between
backends from names alone; it needs their **cost, speed and expected quality**
as data. Each provider therefore publishes:

* ``cost_per_audio_minute_usd`` / token prices -- what a request will cost
* ``typical_rtf`` -- real-time factor, processing seconds per audio second.
  Below 1.0 is faster than realtime; a streaming session needs headroom well
  under 1.0 or its backlog grows without bound.
* ``expected_error_rate`` -- a per-language quality prior, not a promise.

That last field is the honest one. It starts as a documented prior and is
**overwritten by measured values from the eval harness**, so routing decisions
improve as real data arrives rather than staying frozen at whatever a vendor's
marketing page claimed. A provider with no measurement yet is used cautiously;
one with a measured regression is routed away from automatically.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from koe.domain.audio import AudioChunk
from koe.domain.transcript import Diarization, Segment, Transcript
from koe.text.script import Language


class Modality(StrEnum):
    ASR = "asr"
    DIARIZATION = "diarization"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Everything the router needs to know about a backend."""

    name: str
    modality: Modality
    model: str
    languages: frozenset[Language] = field(default_factory=lambda: frozenset({Language.EN}))

    supports_streaming: bool = False
    supports_word_timestamps: bool = False
    supports_language_hint: bool = True

    # cost
    cost_per_audio_minute_usd: float = 0.0
    cost_per_1k_input_tokens_usd: float = 0.0
    cost_per_1k_output_tokens_usd: float = 0.0

    # speed
    typical_rtf: float = 0.0
    typical_first_result_ms: float = 0.0

    #: Per-language error-rate prior (CER for JA, WER for EN). Replaced by
    #: measured values once the eval harness has run against this provider.
    expected_error_rate: dict[Language, float] = field(default_factory=dict)
    #: True once `expected_error_rate` reflects measurement rather than a prior.
    measured: bool = False

    def supports(self, language: Language) -> bool:
        if language in (Language.UNKNOWN, Language.MIXED):
            # mixed audio needs a backend fluent in both
            return {Language.JA, Language.EN} <= self.languages
        return language in self.languages

    def error_rate_for(self, language: Language) -> float:
        """Expected error rate, falling back to the worst known language."""
        if language in self.expected_error_rate:
            return self.expected_error_rate[language]
        if self.expected_error_rate:
            return max(self.expected_error_rate.values())
        return 1.0  # unknown quality is treated as bad, not as free

    def estimate_audio_cost(self, seconds: float) -> float:
        return (seconds / 60.0) * self.cost_per_audio_minute_usd

    def estimate_token_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000.0 * self.cost_per_1k_input_tokens_usd
            + output_tokens / 1000.0 * self.cost_per_1k_output_tokens_usd
        )

    def with_measurement(self, language: Language, error_rate: float) -> ProviderInfo:
        """Return a copy with a measured error rate folded in."""
        updated = dict(self.expected_error_rate)
        updated[language] = error_rate
        return replace(self, expected_error_rate=updated, measured=True)

    def __str__(self) -> str:
        return f"{self.name}/{self.model}"


@dataclass(slots=True)
class Usage:
    """What a single provider call actually consumed."""

    provider: str = ""
    model: str = ""
    audio_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            provider=self.provider or other.provider,
            model=self.model or other.model,
            audio_seconds=self.audio_seconds + other.audio_seconds,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_usd=self.cost_usd + other.cost_usd,
            cached=self.cached and other.cached,
        )


@dataclass(slots=True)
class LLMResponse:
    """A completion plus the accounting for it."""

    text: str
    usage: Usage
    model: str = ""
    stop_reason: str = ""
    raw: Any = None


@dataclass(slots=True)
class Message:
    """One turn in an LLM conversation."""

    role: str
    content: str


class ProviderError(Exception):
    """A provider call failed.

    `retryable` drives the router's fallback logic: a rate limit or timeout is
    worth retrying elsewhere, a malformed request is not and would only burn
    the budget again on the next backend.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        retryable: bool = True,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status = status


class ProviderTimeout(ProviderError):
    """A provider exceeded its deadline."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=True)


@runtime_checkable
class ASRProvider(Protocol):
    """Speech recognition."""

    info: ProviderInfo

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """Transcribe a complete span of audio."""
        ...


@runtime_checkable
class StreamingASRProvider(ASRProvider, Protocol):
    """Speech recognition that emits partial hypotheses while audio arrives."""

    def stream(
        self,
        audio: AsyncIterator[AudioChunk],
        *,
        language: Language | None = None,
    ) -> AsyncIterator[Segment]:
        """Yield segments as they are recognized, partials included."""
        ...


@runtime_checkable
class DiarizationProvider(Protocol):
    """Speaker segmentation."""

    info: ProviderInfo

    async def diarize(
        self,
        audio: AudioChunk,
        *,
        num_speakers: int | None = None,
    ) -> Diarization:
        """Determine who spoke when."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation."""

    info: ProviderInfo

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion."""
        ...


@asynccontextmanager
async def measured(info: ProviderInfo, usage: Usage) -> AsyncIterator[Usage]:
    """Time a provider call and record the latency on `usage`.

    Latency is measured around the call itself rather than reported by the
    provider, because what the router needs to know is the latency the *user*
    experienced -- which includes queueing, transport and retries that a
    vendor's own timing conveniently excludes.
    """
    started = time.perf_counter()
    usage.provider = info.name
    usage.model = info.model
    try:
        yield usage
    finally:
        usage.latency_ms = (time.perf_counter() - started) * 1000.0
