"""Routing: constraint filtering, ranking, fallback, and breaker interaction."""

from __future__ import annotations

import pytest

from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.providers.base import ProviderError
from koe.providers.mock import MockASR
from koe.routing.breaker import BreakerState, CircuitBreaker
from koe.routing.budget import Budget, Priority
from koe.routing.router import NoProviderAvailable, Router
from koe.text.script import Language


def audio(seconds: float = 4.0) -> AudioChunk:
    return AudioChunk(data=b"\x00" * STANDARD_FORMAT.bytes_for(seconds))


def provider(
    name: str,
    *,
    degradation: float = 0.05,
    cost: float = 0.006,
    latency_ms: float = 200.0,
    **kwargs: object,
) -> MockASR:
    return MockASR(
        name=name,
        degradation=degradation,
        cost_per_audio_minute_usd=cost,
        latency_ms=latency_ms,
        **kwargs,  # type: ignore[arg-type]
    )


async def transcribe(p: MockASR) -> object:
    return await p.transcribe(audio())


# --------------------------------------------------------------------------
# hard constraints eliminate rather than penalize
# --------------------------------------------------------------------------


def test_latency_ceiling_excludes_a_slow_provider() -> None:
    """A 200ms cap is not a preference a cheap backend can outweigh."""
    router = Router([provider("fast", latency_ms=100), provider("slow", latency_ms=5_000)])

    decision = router.select(Budget(max_latency_ms=500))

    assert decision.chosen_name == "fast"
    assert "slow" in decision.rejected
    assert "exceeds" in decision.rejected["slow"]


def test_cost_ceiling_excludes_an_expensive_provider() -> None:
    router = Router([provider("cheap", cost=0.001), provider("premium", cost=0.05)])
    decision = router.select(Budget(max_cost_per_audio_minute_usd=0.01))
    assert decision.chosen_name == "cheap"


def test_error_rate_ceiling_excludes_a_bad_provider() -> None:
    router = Router([provider("good", degradation=0.05), provider("bad", degradation=0.40)])
    decision = router.select(Budget(max_error_rate=0.10, language=Language.JA))
    assert decision.chosen_name == "good"
    assert "expected error" in decision.rejected["bad"]


def test_word_timestamp_requirement_is_enforced() -> None:
    router = Router([provider("with_words"), provider("no_words", word_timestamps=False)])
    decision = router.select(Budget(require_word_timestamps=True))
    assert decision.chosen_name == "with_words"


def test_explicit_exclusion() -> None:
    """Operators need a way to pull a provider during an incident."""
    router = Router([provider("a"), provider("b")])
    decision = router.select(Budget(exclude=frozenset({"a"})))
    assert decision.chosen_name == "b"


def test_no_eligible_provider_explains_every_rejection() -> None:
    """'Routing failed' with no reason is impossible to debug at 3am."""
    router = Router([provider("slow", latency_ms=9_000), provider("pricey", cost=1.0)])

    with pytest.raises(NoProviderAvailable) as excinfo:
        router.select(Budget(max_latency_ms=100, max_cost_per_audio_minute_usd=0.001))

    assert set(excinfo.value.rejections) == {"slow", "pricey"}
    assert not excinfo.value.retryable


def test_empty_router() -> None:
    with pytest.raises(NoProviderAvailable):
        Router().select(Budget())


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_latency_priority_prefers_the_fast_provider() -> None:
    router = Router(
        [
            provider("fast_meh", latency_ms=50, degradation=0.15, cost=0.01),
            provider("slow_good", latency_ms=2_000, degradation=0.02, cost=0.01),
        ]
    )
    decision = router.select(Budget(priority=Priority.LATENCY, language=Language.JA))
    assert decision.chosen_name == "fast_meh"


def test_quality_priority_prefers_the_accurate_provider() -> None:
    router = Router(
        [
            provider("fast_meh", latency_ms=50, degradation=0.15, cost=0.01),
            provider("slow_good", latency_ms=2_000, degradation=0.02, cost=0.01),
        ]
    )
    decision = router.select(Budget(priority=Priority.QUALITY, language=Language.JA))
    assert decision.chosen_name == "slow_good"


def test_cost_priority_prefers_the_cheap_provider() -> None:
    router = Router(
        [
            provider("cheap", cost=0.001, degradation=0.12, latency_ms=400),
            provider("premium", cost=0.05, degradation=0.02, latency_ms=400),
        ]
    )
    decision = router.select(Budget(priority=Priority.COST, language=Language.JA))
    assert decision.chosen_name == "cheap"


def test_a_dimension_where_everyone_is_equal_stops_mattering() -> None:
    """Normalizing within the candidate set makes the decision about the
    choice actually available."""
    router = Router(
        [
            provider("a", cost=0.006, degradation=0.02, latency_ms=100),
            provider("b", cost=0.006, degradation=0.20, latency_ms=100),
        ]
    )
    decision = router.select(Budget(priority=Priority.COST, language=Language.JA))
    # cost is identical, so quality decides even under a cost priority
    assert decision.chosen_name == "a"
    assert all(c.cost_score == 0.0 for c in decision.ranked)


def test_selection_is_reproducible() -> None:
    router = Router([provider("a"), provider("b"), provider("c")])
    budget = Budget(language=Language.JA)
    first = [c.name for c in router.select(budget).ranked]
    second = [c.name for c in router.select(budget).ranked]
    assert first == second


def test_decision_explains_itself() -> None:
    router = Router([provider("a"), provider("slow", latency_ms=9_000)])
    text = router.select(Budget(max_latency_ms=1_000)).explain()
    assert "rejected" in text
    assert "score=" in text


def test_preset_budgets() -> None:
    assert Budget.realtime().require_streaming
    assert Budget.accurate().priority is Priority.QUALITY
    assert Budget.bulk().priority is Priority.COST


# --------------------------------------------------------------------------
# execution and fallback
# --------------------------------------------------------------------------


async def test_a_healthy_primary_is_used_without_fallback() -> None:
    router = Router([provider("primary", latency_ms=0), provider("backup", latency_ms=0)])
    result = await router.execute(Budget(priority=Priority.QUALITY), transcribe)
    assert not result.fell_back
    assert len(result.attempts) == 1


async def test_a_retryable_failure_falls_back() -> None:
    # "broken" has to be genuinely preferred, or the router picks the backup
    # first and never exercises the fallback path at all.
    router = Router(
        [
            provider("broken", always_fail=True, latency_ms=0, degradation=0.0),
            provider("backup", latency_ms=0, degradation=0.20),
        ]
    )
    assert router.select(Budget(priority=Priority.QUALITY, language=Language.JA)).chosen_name == (
        "broken"
    )

    result = await router.execute(
        Budget(priority=Priority.QUALITY, language=Language.JA), transcribe
    )

    assert result.provider == "backup"
    assert result.fell_back
    assert result.attempts[0].error is not None


async def test_a_non_retryable_failure_stops_the_chain() -> None:
    """A malformed request fails identically at the next provider."""

    async def bad_request(p: MockASR) -> object:
        raise ProviderError("invalid audio encoding", provider=p.info.name, retryable=False)

    router = Router([provider("a", latency_ms=0), provider("b", latency_ms=0)])

    with pytest.raises(ProviderError) as excinfo:
        await router.execute(Budget(), bad_request)

    assert not excinfo.value.retryable
    assert router.breaker("a").state is BreakerState.CLOSED  # health untouched


async def test_all_providers_failing_raises_with_the_reasons() -> None:
    router = Router(
        [
            provider("a", always_fail=True, latency_ms=0),
            provider("b", always_fail=True, latency_ms=0),
        ]
    )

    with pytest.raises(NoProviderAvailable) as excinfo:
        await router.execute(Budget(), transcribe)

    assert set(excinfo.value.rejections) == {"a", "b"}


async def test_max_attempts_bounds_the_chain() -> None:
    router = Router([provider(f"p{i}", always_fail=True, latency_ms=0) for i in range(5)])

    with pytest.raises(NoProviderAvailable):
        await router.execute(Budget(), transcribe, max_attempts=2)

    attempted = sum(1 for b in router.health() if b["total_failures"])
    assert attempted == 2


# --------------------------------------------------------------------------
# breaker integration
# --------------------------------------------------------------------------


async def test_repeated_failures_open_the_breaker_and_skip_the_provider() -> None:
    router = Router(
        [
            provider("flaky", always_fail=True, latency_ms=0),
            provider("stable", latency_ms=0),
        ],
        failure_threshold=2,
    )

    for _ in range(3):
        await router.execute(Budget(), transcribe)

    assert router.breaker("flaky").state is BreakerState.OPEN
    # once open it is filtered out during selection, not merely failed against
    decision = router.select(Budget())
    assert "flaky" in decision.rejected
    assert "circuit breaker" in decision.rejected["flaky"]


async def test_a_non_retryable_error_does_not_open_the_breaker() -> None:
    """A bad request says nothing about provider health."""

    async def bad_request(p: MockASR) -> object:
        raise ProviderError("bad input", provider=p.info.name, retryable=False)

    router = Router([provider("a", latency_ms=0)], failure_threshold=1)

    for _ in range(3):
        with pytest.raises(ProviderError):
            await router.execute(Budget(), bad_request)

    assert router.breaker("a").state is BreakerState.CLOSED


# --------------------------------------------------------------------------
# feedback loop from evaluation
# --------------------------------------------------------------------------


def test_a_measured_error_rate_changes_the_routing_decision() -> None:
    """Routing starts from priors and converges on measurement."""
    router = Router(
        [
            provider("vendor_a", degradation=0.03, cost=0.006, latency_ms=300),
            provider("vendor_b", degradation=0.06, cost=0.006, latency_ms=300),
        ]
    )
    budget = Budget(priority=Priority.QUALITY, language=Language.JA)
    assert router.select(budget).chosen_name == "vendor_a"

    # production measurement says vendor_a is actually much worse on Japanese
    router.record_measurement("vendor_a", Language.JA, 0.22)

    assert router.select(budget).chosen_name == "vendor_b"


# --------------------------------------------------------------------------
# breaker unit behaviour
# --------------------------------------------------------------------------


def test_breaker_half_opens_after_cooldown_and_admits_one_probe() -> None:
    now = [0.0]
    breaker = CircuitBreaker(
        name="p", failure_threshold=2, cooldown_seconds=10.0, clock=lambda: now[0]
    )

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.acquire()

    now[0] = 11.0
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.acquire()  # first probe admitted
    assert not breaker.acquire()  # a recovering service is not flooded

    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_the_breaker() -> None:
    now = [0.0]
    breaker = CircuitBreaker(
        name="p", failure_threshold=1, cooldown_seconds=5.0, clock=lambda: now[0]
    )
    breaker.record_failure()
    now[0] = 6.0
    assert breaker.acquire()

    breaker.record_failure()

    assert breaker.state is BreakerState.OPEN


def test_breaker_reset_is_available_for_operators() -> None:
    breaker = CircuitBreaker(name="p", failure_threshold=1)
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    breaker.reset()
    assert breaker.state is BreakerState.CLOSED
