"""Service-level budgets: the constraints a routing decision has to satisfy.

Every request in a voice product is a three-way trade between latency, cost and
quality, and the right answer differs per request rather than per deployment.
Live captions during a meeting need a first result inside a few hundred
milliseconds and will accept a worse transcript to get it. The same audio
re-processed overnight for the archived 議事録 has no latency constraint at all
and should use the best model available. Batch re-processing of a year of
recordings is dominated by cost.

Encoding that as a :class:`Budget` -- passed per request -- keeps the trade-off
explicit and reviewable, instead of scattered through call sites as hardcoded
model names that nobody remembers the reason for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from koe.text.script import Language


class Priority(StrEnum):
    """What to optimize when constraints leave room to choose."""

    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    BALANCED = "balanced"

    @property
    def weights(self) -> tuple[float, float, float]:
        """``(quality, cost, latency)`` weights, summing to 1."""
        return {
            Priority.LATENCY: (0.2, 0.1, 0.7),
            Priority.COST: (0.25, 0.6, 0.15),
            Priority.QUALITY: (0.7, 0.1, 0.2),
            Priority.BALANCED: (0.45, 0.25, 0.30),
        }[self]


@dataclass(frozen=True, slots=True)
class Budget:
    """Constraints and preferences for one request.

    `max_*` fields are **hard constraints**: a provider that cannot meet them is
    excluded outright rather than penalized, because a 200 ms cap on live
    captions is not a preference that a very cheap model can outweigh.
    `priority` then orders whatever survives.
    """

    priority: Priority = Priority.BALANCED
    max_latency_ms: float | None = None
    max_cost_per_audio_minute_usd: float | None = None
    max_error_rate: float | None = None
    require_streaming: bool = False
    require_word_timestamps: bool = False
    language: Language = Language.UNKNOWN
    #: Providers to exclude by name, e.g. during an incident.
    exclude: frozenset[str] = frozenset()

    @classmethod
    def realtime(cls, language: Language = Language.UNKNOWN) -> Budget:
        """Live captioning: a late transcript is a wrong transcript."""
        return cls(
            priority=Priority.LATENCY,
            max_latency_ms=800.0,
            require_streaming=True,
            language=language,
        )

    @classmethod
    def accurate(cls, language: Language = Language.UNKNOWN) -> Budget:
        """Post-meeting 議事録: no latency constraint, use the best model."""
        return cls(
            priority=Priority.QUALITY,
            require_word_timestamps=True,
            language=language,
        )

    @classmethod
    def bulk(cls, language: Language = Language.UNKNOWN) -> Budget:
        """Archive re-processing: dominated by cost per hour of audio."""
        return cls(
            priority=Priority.COST,
            max_error_rate=0.25,
            language=language,
        )

    def __str__(self) -> str:
        parts = [self.priority.value]
        if self.max_latency_ms is not None:
            parts.append(f"<{self.max_latency_ms:.0f}ms")
        if self.max_cost_per_audio_minute_usd is not None:
            parts.append(f"<${self.max_cost_per_audio_minute_usd:.4f}/min")
        if self.max_error_rate is not None:
            parts.append(f"<{self.max_error_rate:.0%} err")
        if self.require_streaming:
            parts.append("streaming")
        return f"Budget({', '.join(parts)})"
