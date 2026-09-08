"""Budget-aware provider selection with fallback.

The router answers "which backend should serve this request" from data rather
than from a hardcoded default, in two stages:

**Filter on hard constraints.** Language support, streaming, word timestamps, a
latency ceiling, a cost ceiling, an error-rate ceiling, and circuit-breaker
health. These are eliminations, not penalties -- a 200 ms cap on live captions
is not a preference that a very cheap backend can outweigh by being cheap.

**Rank whatever survives.** Each remaining candidate is scored on quality, cost
and latency, normalized across the candidate set so the three commensurate, and
weighted by the request's :class:`~koe.routing.budget.Priority`.

Normalizing *within the candidate set* rather than against absolute scales is
deliberate: it makes the decision about the choice actually available. If every
candidate costs about the same, cost stops influencing the ranking and quality
and latency decide, which is the behaviour you want and not what fixed scales
give you.

Execution then walks the ranking. A retryable failure moves to the next
candidate; a non-retryable one stops immediately, because a malformed request
will be malformed at the next provider too and trying again only spends the
latency budget twice.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from koe.providers.base import ProviderError, ProviderInfo
from koe.routing.breaker import CircuitBreaker
from koe.routing.budget import Budget
from koe.text.script import Language

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class NoProviderAvailable(ProviderError):
    """No registered provider satisfied the budget.

    Carries the per-provider rejection reasons, because "routing failed" with
    no explanation is impossible to debug at 3am -- the useful question is
    always *which* constraint eliminated the backend you expected.
    """

    def __init__(self, budget: Budget, rejections: dict[str, str]) -> None:
        self.budget = budget
        self.rejections = rejections
        detail = "; ".join(f"{name}: {reason}" for name, reason in sorted(rejections.items()))
        super().__init__(
            f"no provider satisfies {budget} ({detail or 'none registered'})",
            retryable=False,
        )


@dataclass(slots=True)
class Candidate(Generic[T]):
    """A provider that passed the filters, with its score."""

    provider: T
    info: ProviderInfo
    score: float = 0.0
    quality_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0

    @property
    def name(self) -> str:
        return self.info.name


@dataclass(slots=True)
class RoutingDecision(Generic[T]):
    """The outcome of a selection, kept for logging and debugging."""

    budget: Budget
    ranked: list[Candidate[T]]
    rejected: dict[str, str] = field(default_factory=dict)

    @property
    def chosen(self) -> T:
        return self.ranked[0].provider

    @property
    def chosen_name(self) -> str:
        return self.ranked[0].name

    def explain(self) -> str:
        lines = [f"budget: {self.budget}"]
        for rank, candidate in enumerate(self.ranked, start=1):
            lines.append(
                f"  {rank}. {candidate.name:<20} score={candidate.score:.3f} "
                f"(q={candidate.quality_score:.2f} c={candidate.cost_score:.2f} "
                f"l={candidate.latency_score:.2f})"
            )
        for name, reason in sorted(self.rejected.items()):
            lines.append(f"  -- {name:<20} rejected: {reason}")
        return "\n".join(lines)


def _normalize(values: Sequence[float]) -> list[float]:
    """Min-max normalize to [0, 1]; all-equal inputs collapse to 0.

    Collapsing rather than spreading is the point: when every candidate costs
    the same, cost should not influence the ranking at all.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


@dataclass(slots=True)
class Attempt:
    """One provider call inside a routed execution."""

    provider: str
    ok: bool
    latency_ms: float
    error: str | None = None


@dataclass(slots=True)
class RoutedResult(Generic[R]):
    """A successful result plus the path taken to get it."""

    value: R
    provider: str
    attempts: list[Attempt]
    decision: RoutingDecision[Any]

    @property
    def fell_back(self) -> bool:
        return len(self.attempts) > 1

    @property
    def total_latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)


class Router(Generic[T]):
    """Selects and calls providers according to a budget."""

    def __init__(
        self,
        providers: Iterable[T] = (),
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._providers: dict[str, T] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        for provider in providers:
            self.register(provider)

    # -- registration --------------------------------------------------------

    def register(self, provider: T) -> None:
        info: ProviderInfo = provider.info  # type: ignore[attr-defined]
        self._providers[info.name] = provider
        self._breakers[info.name] = CircuitBreaker(
            name=info.name,
            failure_threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown_seconds,
            clock=self._clock,
        )

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)
        self._breakers.pop(name, None)

    @property
    def names(self) -> list[str]:
        return sorted(self._providers)

    def breaker(self, name: str) -> CircuitBreaker:
        return self._breakers[name]

    def health(self) -> list[dict[str, object]]:
        return [b.snapshot() for b in self._breakers.values()]

    # -- feedback ------------------------------------------------------------

    def record_measurement(self, name: str, language: Language, error_rate: float) -> None:
        """Fold a measured error rate into a provider's routing metadata.

        This is how the eval harness closes the loop: routing starts from
        documented priors and converges on measured reality, so a provider that
        regresses in production is routed away from without anyone editing a
        constant.
        """
        provider = self._providers.get(name)
        if provider is None:
            return
        provider.info = provider.info.with_measurement(language, error_rate)  # type: ignore[attr-defined]

    # -- selection -----------------------------------------------------------

    def _reject(self, info: ProviderInfo, budget: Budget) -> str | None:
        """Return a rejection reason, or None if the provider is eligible."""
        if info.name in budget.exclude:
            return "excluded by request"
        if budget.language is not Language.UNKNOWN and not info.supports(budget.language):
            return f"does not support {budget.language.value}"
        if budget.require_streaming and not info.supports_streaming:
            return "does not support streaming"
        if budget.require_word_timestamps and not info.supports_word_timestamps:
            return "does not provide word timestamps"
        if (
            budget.max_latency_ms is not None
            and info.typical_first_result_ms > budget.max_latency_ms
        ):
            return (
                f"typical latency {info.typical_first_result_ms:.0f}ms "
                f"exceeds {budget.max_latency_ms:.0f}ms"
            )
        if (
            budget.max_cost_per_audio_minute_usd is not None
            and info.cost_per_audio_minute_usd > budget.max_cost_per_audio_minute_usd
        ):
            return (
                f"cost ${info.cost_per_audio_minute_usd:.4f}/min exceeds "
                f"${budget.max_cost_per_audio_minute_usd:.4f}/min"
            )
        if budget.max_error_rate is not None:
            expected = info.error_rate_for(budget.language)
            if expected > budget.max_error_rate:
                return f"expected error {expected:.1%} exceeds {budget.max_error_rate:.1%}"
        breaker = self._breakers.get(info.name)
        if breaker is not None and not breaker.is_available:
            return f"circuit breaker {breaker.state.value}"
        return None

    def select(self, budget: Budget) -> RoutingDecision[T]:
        """Rank eligible providers for `budget`.

        Raises :class:`NoProviderAvailable` when nothing qualifies.
        """
        eligible: list[Candidate[T]] = []
        rejected: dict[str, str] = {}

        for name, provider in self._providers.items():
            info: ProviderInfo = provider.info  # type: ignore[attr-defined]
            reason = self._reject(info, budget)
            if reason is not None:
                rejected[name] = reason
                continue
            eligible.append(Candidate(provider=provider, info=info))

        if not eligible:
            raise NoProviderAvailable(budget, rejected)

        quality = _normalize([c.info.error_rate_for(budget.language) for c in eligible])
        cost = _normalize([c.info.cost_per_audio_minute_usd for c in eligible])
        latency = _normalize([c.info.typical_first_result_ms for c in eligible])
        w_quality, w_cost, w_latency = budget.priority.weights

        for candidate, q, c, latency_score in zip(eligible, quality, cost, latency, strict=True):
            candidate.quality_score = q
            candidate.cost_score = c
            candidate.latency_score = latency_score
            candidate.score = w_quality * q + w_cost * c + w_latency * latency_score

        # lower is better; ties break on name so a decision is reproducible
        eligible.sort(key=lambda c: (c.score, c.name))
        return RoutingDecision(budget=budget, ranked=eligible, rejected=rejected)

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        budget: Budget,
        call: Callable[[T], Awaitable[R]],
        *,
        max_attempts: int = 3,
        total_deadline_ms: float | None = None,
    ) -> RoutedResult[R]:
        """Run `call` against the best provider, falling back on failure.

        `total_deadline_ms` bounds the *whole* chain rather than each attempt.
        Per-attempt deadlines are how a three-provider fallback quietly turns a
        1-second budget into 3 seconds of user-visible latency.
        """
        decision = self.select(budget)
        attempts: list[Attempt] = []
        started = time.perf_counter()
        last_error: ProviderError | None = None

        for candidate in decision.ranked[:max_attempts]:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if total_deadline_ms is not None and elapsed_ms >= total_deadline_ms:
                logger.warning(
                    "routing deadline %.0fms exhausted after %d attempt(s)",
                    total_deadline_ms,
                    len(attempts),
                )
                break

            breaker = self._breakers[candidate.name]
            if not breaker.acquire():
                attempts.append(
                    Attempt(candidate.name, ok=False, latency_ms=0.0, error="breaker open")
                )
                continue

            attempt_started = time.perf_counter()
            try:
                value = await call(candidate.provider)
            except ProviderError as exc:
                latency_ms = (time.perf_counter() - attempt_started) * 1000.0
                breaker.record_failure(retryable=exc.retryable)
                attempts.append(
                    Attempt(candidate.name, ok=False, latency_ms=latency_ms, error=str(exc))
                )
                last_error = exc
                if not exc.retryable:
                    # This request is broken, not the provider. Trying the next
                    # backend would fail identically and spend the budget twice.
                    logger.info("not retrying %s: non-retryable error", candidate.name)
                    raise
                logger.info("provider %s failed, falling back: %s", candidate.name, exc)
                continue
            except Exception as exc:
                latency_ms = (time.perf_counter() - attempt_started) * 1000.0
                breaker.record_failure(retryable=True)
                attempts.append(
                    Attempt(candidate.name, ok=False, latency_ms=latency_ms, error=repr(exc))
                )
                raise

            latency_ms = (time.perf_counter() - attempt_started) * 1000.0
            breaker.record_success()
            attempts.append(Attempt(candidate.name, ok=True, latency_ms=latency_ms))
            return RoutedResult(
                value=value,
                provider=candidate.name,
                attempts=attempts,
                decision=decision,
            )

        raise NoProviderAvailable(
            budget,
            {a.provider: a.error or "failed" for a in attempts}
            or {"(none)": str(last_error) if last_error else "no candidates attempted"},
        )
