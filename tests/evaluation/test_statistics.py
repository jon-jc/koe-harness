"""Bootstrap intervals and paired significance testing."""

from __future__ import annotations

import pytest

from koe.evaluation.metrics import ErrorRate, Unit, word_error_rate
from koe.evaluation.statistics import (
    bootstrap_interval,
    paired_bootstrap,
    paired_permutation_test,
    summarize,
)


def rate(errors: int, length: int) -> ErrorRate:
    return ErrorRate(
        substitutions=errors,
        deletions=0,
        insertions=0,
        reference_length=length,
        unit=Unit.WORD,
        tokenizer="whitespace",
    )


# --------------------------------------------------------------------------
# bootstrap intervals
# --------------------------------------------------------------------------


def test_interval_brackets_the_point_estimate() -> None:
    samples = [rate(i % 3, 10) for i in range(40)]
    interval = bootstrap_interval(samples, iterations=500)
    assert interval.lower <= interval.point <= interval.upper


def test_results_are_reproducible() -> None:
    """Re-running an eval must not move the interval, or reviewers cannot
    distinguish a real change from resampling jitter."""
    samples = [rate(i % 4, 12) for i in range(30)]
    first = bootstrap_interval(samples, iterations=400)
    second = bootstrap_interval(samples, iterations=400)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_a_different_seed_gives_a_different_but_similar_interval() -> None:
    samples = [rate(i % 4, 12) for i in range(30)]
    a = bootstrap_interval(samples, iterations=400, seed=1)
    b = bootstrap_interval(samples, iterations=400, seed=2)
    assert (a.lower, a.upper) != (b.lower, b.upper)
    assert a.point == b.point  # the point estimate is not resampled


def test_more_data_narrows_the_interval() -> None:
    small = bootstrap_interval([rate(i % 3, 10) for i in range(8)], iterations=800)
    large = bootstrap_interval([rate(i % 3, 10) for i in range(200)], iterations=800)
    assert large.width < small.width


def test_a_single_sample_yields_a_degenerate_interval() -> None:
    """One utterance carries no information about between-utterance variance."""
    interval = bootstrap_interval([rate(1, 10)])
    assert interval.lower == interval.upper == interval.point


def test_no_samples() -> None:
    assert bootstrap_interval([]).point == 0.0


def test_pooling_not_averaging() -> None:
    """A short utterance must not count as much as a long one."""
    samples = [rate(1, 2), rate(0, 98)]
    assert bootstrap_interval(samples).point == pytest.approx(0.01)


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------


def test_identical_systems_are_inconclusive() -> None:
    samples = [rate(i % 3, 10) for i in range(30)]
    result = paired_bootstrap(samples, samples, iterations=400)
    assert result.delta == pytest.approx(0.0)
    assert not result.significant
    assert result.verdict == "inconclusive"


def test_a_large_consistent_improvement_is_significant() -> None:
    baseline = [rate(4, 10) for _ in range(40)]
    candidate = [rate(1, 10) for _ in range(40)]

    result = paired_bootstrap(baseline, candidate, iterations=600)

    assert result.delta < 0
    assert result.significant
    assert result.verdict == "improvement"


def test_a_large_consistent_regression_is_flagged() -> None:
    baseline = [rate(1, 10) for _ in range(40)]
    candidate = [rate(4, 10) for _ in range(40)]

    result = paired_bootstrap(baseline, candidate, iterations=600)

    assert result.verdict == "regression"
    assert result.significant


def test_a_tiny_difference_on_a_small_set_is_not_significant() -> None:
    """The case that matters: noise must not read as an improvement."""
    baseline = [rate(2, 10) for _ in range(8)]
    candidate = [rate(2, 10) for _ in range(7)] + [rate(1, 10)]

    result = paired_bootstrap(baseline, candidate, iterations=600)

    assert result.delta < 0  # candidate looks better
    assert not result.significant  # but not distinguishably so


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap([rate(1, 10)], [rate(1, 10), rate(2, 10)])


def test_mixed_units_are_rejected() -> None:
    char_sample = ErrorRate(1, 0, 0, 10, Unit.CHARACTER)
    with pytest.raises(ValueError, match="units"):
        paired_bootstrap([rate(1, 10)], [char_sample])


def test_relative_delta() -> None:
    baseline = [rate(4, 10) for _ in range(20)]
    candidate = [rate(2, 10) for _ in range(20)]
    result = paired_bootstrap(baseline, candidate, iterations=200)
    assert result.relative_delta == pytest.approx(-0.5)


# --------------------------------------------------------------------------
# permutation test
# --------------------------------------------------------------------------


def test_p_value_is_never_exactly_zero() -> None:
    """Finite permutations cannot support a claim of absolute certainty."""
    baseline = [rate(9, 10) for _ in range(50)]
    candidate = [rate(0, 10) for _ in range(50)]
    p = paired_permutation_test(baseline, candidate, iterations=200)
    assert p > 0.0
    assert p == pytest.approx(1 / 201)


def test_identical_systems_give_a_high_p_value() -> None:
    samples = [rate(i % 3, 10) for i in range(30)]
    assert paired_permutation_test(samples, samples, iterations=300) > 0.5


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------


def test_summary_carries_the_tokenizer() -> None:
    samples = [word_error_rate("the cat sat", "the cat sit") for _ in range(10)]
    summary = summarize("WER", samples, iterations=200)
    assert summary.tokenizer == "whitespace"
    assert summary.n_samples == 10
    assert "WER" in str(summary)
