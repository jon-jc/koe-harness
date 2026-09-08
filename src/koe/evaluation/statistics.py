"""Uncertainty quantification for evaluation results.

An error rate computed on a 40-utterance eval set is an estimate, not a
measurement. Reporting "CER improved from 8.1% to 7.6%" without saying whether
that could be noise is how teams ship regressions while believing they shipped
improvements -- and with small eval sets, which is what everyone actually has,
it usually *is* noise.

Two techniques, both non-parametric because error rates are bounded, skewed,
and nothing like normally distributed:

**Bootstrap confidence intervals** for "what is this system's error rate,
give or take?"

**Paired bootstrap and paired permutation** for "is B actually better than A?"
Pairing is the important part. Both systems are scored on the same utterances,
so resampling *the same indices for both* cancels between-utterance difficulty:
the hard utterances are hard for both systems, and an unpaired test would let
that shared variance swamp the difference being measured. Pairing routinely
turns an inconclusive comparison into a decisive one without touching the data.

**Resampling happens at the utterance level, never the character level.**
Errors inside one utterance are strongly correlated -- a model that loses the
audio mid-sentence gets the whole clause wrong. Treating characters as
independent samples would understate the true variance by a large factor and
produce confidence intervals far too narrow to be honest.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from koe.evaluation.metrics import ErrorRate, Unit

DEFAULT_ITERATIONS = 2_000
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate with an interval around it."""

    point: float
    lower: float
    upper: float
    level: float = DEFAULT_LEVEL

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0

    def __str__(self) -> str:
        return f"{self.point:.2%} [{self.lower:.2%}, {self.upper:.2%}] @{self.level:.0%}"


def _pool(samples: Sequence[ErrorRate]) -> float:
    """Corpus error rate from per-utterance results.

    Totals are pooled rather than averaged: a mean of per-utterance rates
    weights a two-word utterance the same as a two-hundred-word one, so it
    shifts when segmentation changes even though the audio did not.
    """
    errors = 0
    length = 0
    for sample in samples:
        errors += sample.errors
        length += sample.reference_length
    if length == 0:
        return 0.0
    return errors / length


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of a sorted-in-place list."""
    if not values:
        return 0.0
    values.sort()
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[int(position)]
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_interval(
    samples: Sequence[ErrorRate],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for a pooled error rate.

    `seed` is fixed by default so a report is reproducible: re-running an
    evaluation must not move the interval, or reviewers cannot tell a real
    change from resampling jitter.
    """
    point = _pool(samples)
    if len(samples) < 2:
        # A single utterance carries no information about between-utterance
        # variance, so the honest interval is degenerate rather than invented.
        return ConfidenceInterval(point=point, lower=point, upper=point, level=level)

    rng = random.Random(seed)
    n = len(samples)
    replicates: list[float] = []
    for _ in range(iterations):
        drawn = [samples[rng.randrange(n)] for _ in range(n)]
        replicates.append(_pool(drawn))

    tail = (1.0 - level) / 2.0
    return ConfidenceInterval(
        point=point,
        lower=_percentile(replicates, tail),
        upper=_percentile(replicates, 1.0 - tail),
        level=level,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """A/B result between two systems scored on the same utterances."""

    baseline: float
    candidate: float
    delta_interval: ConfidenceInterval
    p_value: float
    n: int
    alpha: float = 0.05

    @property
    def delta(self) -> float:
        """Candidate minus baseline. Negative is an improvement for error rates."""
        return self.candidate - self.baseline

    @property
    def relative_delta(self) -> float:
        if self.baseline == 0:
            return 0.0
        return self.delta / self.baseline

    @property
    def significant(self) -> bool:
        """Whether the difference survives both the interval and the permutation test."""
        return self.p_value < self.alpha and self.delta_interval.excludes_zero

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "improvement" if self.delta < 0 else "regression"

    def __str__(self) -> str:
        return (
            f"{self.baseline:.2%} -> {self.candidate:.2%} "
            f"(Δ {self.delta:+.2%}, {self.delta_interval.lower:+.2%}..{self.delta_interval.upper:+.2%}, "
            f"p={self.p_value:.3f}, n={self.n}) {self.verdict}"
        )


def paired_bootstrap(
    baseline: Sequence[ErrorRate],
    candidate: Sequence[ErrorRate],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = DEFAULT_LEVEL,
    alpha: float = 0.05,
    seed: int = 0,
) -> Comparison:
    """Compare two systems on the same utterances.

    Both sequences must be aligned -- index *i* is the same utterance scored by
    each system. Each bootstrap replicate resamples utterance indices once and
    applies them to both systems, which cancels the shared difficulty of the
    utterances and leaves only the difference between the systems.

    The p-value comes from a paired permutation test rather than the bootstrap:
    under the null hypothesis that the systems are equivalent, swapping their
    labels on any utterance changes nothing, so the distribution of deltas over
    random swaps *is* the null distribution. No parametric assumption is needed.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs aligned samples, got {len(baseline)} and {len(candidate)}"
        )
    units = {s.unit for s in (*baseline, *candidate)}
    if len(units) > 1:
        raise ValueError(f"cannot compare across units: {units}")

    n = len(baseline)
    base_point = _pool(baseline)
    cand_point = _pool(candidate)

    if n < 2:
        return Comparison(
            baseline=base_point,
            candidate=cand_point,
            delta_interval=ConfidenceInterval(
                point=cand_point - base_point,
                lower=cand_point - base_point,
                upper=cand_point - base_point,
                level=level,
            ),
            p_value=1.0,
            n=n,
            alpha=alpha,
        )

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        base_draw = [baseline[i] for i in indices]
        cand_draw = [candidate[i] for i in indices]
        deltas.append(_pool(cand_draw) - _pool(base_draw))

    tail = (1.0 - level) / 2.0
    interval = ConfidenceInterval(
        point=cand_point - base_point,
        lower=_percentile(deltas, tail),
        upper=_percentile(deltas, 1.0 - tail),
        level=level,
    )

    p_value = paired_permutation_test(baseline, candidate, iterations=iterations, seed=seed + 1)

    return Comparison(
        baseline=base_point,
        candidate=cand_point,
        delta_interval=interval,
        p_value=p_value,
        n=n,
        alpha=alpha,
    )


def paired_permutation_test(
    baseline: Sequence[ErrorRate],
    candidate: Sequence[ErrorRate],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation p-value for the difference in pooled rates.

    Uses the ``(hits + 1) / (iterations + 1)`` estimator, which cannot return
    exactly zero. A p-value of 0 would claim more certainty than a finite number
    of permutations can support.
    """
    n = len(baseline)
    if n == 0:
        return 1.0

    observed = abs(_pool(candidate) - _pool(baseline))
    rng = random.Random(seed)
    at_least_as_extreme = 0

    for _ in range(iterations):
        left: list[ErrorRate] = []
        right: list[ErrorRate] = []
        for i in range(n):
            if rng.random() < 0.5:
                left.append(baseline[i])
                right.append(candidate[i])
            else:
                left.append(candidate[i])
                right.append(baseline[i])
        if abs(_pool(right) - _pool(left)) >= observed:
            at_least_as_extreme += 1

    return (at_least_as_extreme + 1) / (iterations + 1)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """A pooled metric with its uncertainty, ready to print or persist."""

    name: str
    interval: ConfidenceInterval
    unit: Unit
    tokenizer: str
    n_samples: int

    @property
    def value(self) -> float:
        return self.interval.point

    def __str__(self) -> str:
        suffix = f" [{self.tokenizer}]" if self.tokenizer else ""
        return f"{self.name}{suffix}: {self.interval} (n={self.n_samples})"


def summarize(
    name: str,
    samples: Sequence[ErrorRate],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> MetricSummary:
    """Pool `samples` and attach a bootstrap interval."""
    interval = bootstrap_interval(samples, iterations=iterations, level=level, seed=seed)
    unit = samples[0].unit if samples else Unit.CHARACTER
    tokenizer = next((s.tokenizer for s in samples if s.tokenizer), "")
    return MetricSummary(
        name=name,
        interval=interval,
        unit=unit,
        tokenizer=tokenizer,
        n_samples=len(samples),
    )
