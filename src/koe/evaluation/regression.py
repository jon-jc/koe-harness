"""Regression gates for CI.

A quality gate is only useful if the team leaves it switched on, and the fastest
way to get one switched off is for it to fail on noise. On a forty-case eval
set, run-to-run variation of half a point of CER is entirely ordinary; a gate
with a fixed ``cer < 0.08`` threshold will fire on that, someone will re-run it
until it goes green, and within a month the gate means nothing.

So the default here is: **fail on a regression only when it is statistically
significant.** Candidate and baseline are paired by case id, compared with a
paired bootstrap, and a regression has to clear both the confidence interval
and the permutation test before it blocks a merge. A regression that cannot be
distinguished from noise is reported as a warning and does not fail the build.

Absolute ceilings are still supported (`max_value`) for the cases where the
requirement is contractual rather than comparative -- "we do not ship above 15%
CER" is a product decision, not a statistical one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from koe.evaluation.metrics import ErrorRate
from koe.evaluation.runner import EvalReport
from koe.evaluation.statistics import Comparison, paired_bootstrap


class Metric(StrEnum):
    """Metrics a gate can be defined on."""

    CER = "cer"
    WER = "wer"
    DER = "der"
    FAILURE_RATE = "failure_rate"
    COST_PER_AUDIO_HOUR = "cost_per_audio_hour"
    LATENCY_P95 = "latency_p95_ms"
    RTF = "mean_rtf"


@dataclass(frozen=True, slots=True)
class Gate:
    """One quality condition a candidate must satisfy.

    `max_value` is an absolute ceiling. `max_regression` is how much worse than
    the baseline the candidate may be -- as a fraction of the baseline when
    `relative` is set, otherwise in the metric's own units.
    """

    metric: Metric
    max_value: float | None = None
    max_regression: float | None = None
    relative: bool = True
    require_significance: bool = True
    tag: str | None = None

    @property
    def name(self) -> str:
        return f"{self.metric.value}{f'[{self.tag}]' if self.tag else ''}"


@dataclass(slots=True)
class GateResult:
    """Outcome of evaluating one gate."""

    gate: Gate
    passed: bool
    observed: float
    baseline: float | None = None
    threshold: float | None = None
    comparison: Comparison | None = None
    detail: str = ""
    warning: bool = False

    @property
    def status(self) -> str:
        if self.passed and self.warning:
            return "WARN"
        return "PASS" if self.passed else "FAIL"


@dataclass(slots=True)
class RegressionReport:
    """All gate outcomes for one candidate."""

    system: str
    baseline_system: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.warning and r.passed]

    def render(self) -> str:
        lines = [
            f"regression check: {self.baseline_system} -> {self.system}",
            "-" * 72,
        ]
        for result in self.results:
            lines.append(f"  [{result.status}] {result.gate.name:<22} {result.detail}")
        lines.append("-" * 72)
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def paired_samples(
    baseline: EvalReport,
    candidate: EvalReport,
    *,
    metric: Metric,
    tag: str | None = None,
) -> tuple[list[ErrorRate], list[ErrorRate]]:
    """Align per-case scores by case id.

    Only cases that both systems handled successfully are used. Including a
    case that one system failed on would silently mix a reliability difference
    into a quality comparison, and those need to be reported separately.
    """
    if metric not in (Metric.CER, Metric.WER):
        return ([], [])

    base_by_id = {r.case_id: r for r in baseline.succeeded}
    cand_by_id = {r.case_id: r for r in candidate.succeeded}
    shared = sorted(set(base_by_id) & set(cand_by_id))

    base_samples: list[ErrorRate] = []
    cand_samples: list[ErrorRate] = []
    for case_id in shared:
        base_result = base_by_id[case_id]
        cand_result = cand_by_id[case_id]
        if tag is not None and tag not in cand_result.tags:
            continue
        if base_result.score is None or cand_result.score is None:
            continue
        if metric is Metric.CER:
            base_samples.append(base_result.score.cer)
            cand_samples.append(cand_result.score.cer)
        else:
            base_samples.append(base_result.score.wer)
            cand_samples.append(cand_result.score.wer)
    return (base_samples, cand_samples)


def _observed(report: EvalReport, metric: Metric, tag: str | None) -> float:
    if metric in (Metric.CER, Metric.WER):
        subset = [r for r in report.succeeded if tag in r.tags] if tag else report.succeeded
        samples = [
            (r.score.cer if metric is Metric.CER else r.score.wer) for r in subset if r.score
        ]
        total_errors = sum(s.errors for s in samples)
        total_length = sum(s.reference_length for s in samples)
        return total_errors / total_length if total_length else 0.0
    if metric is Metric.DER:
        ders = [r.diarization.value for r in report.succeeded if r.diarization]
        return sum(ders) / len(ders) if ders else 0.0
    if metric is Metric.FAILURE_RATE:
        return report.failure_rate
    if metric is Metric.COST_PER_AUDIO_HOUR:
        return report.cost_per_audio_hour
    if metric is Metric.LATENCY_P95:
        return report.latency_percentile(0.95)
    return report.mean_rtf


def evaluate_gate(
    gate: Gate,
    candidate: EvalReport,
    baseline: EvalReport | None,
    *,
    seed: int = 0,
) -> GateResult:
    """Check one gate against a candidate, optionally versus a baseline."""
    observed = _observed(candidate, gate.metric, gate.tag)

    # Absolute ceiling: a product requirement, not a statistical claim.
    if gate.max_value is not None and observed > gate.max_value:
        return GateResult(
            gate=gate,
            passed=False,
            observed=observed,
            threshold=gate.max_value,
            detail=f"{observed:.4g} exceeds ceiling {gate.max_value:.4g}",
        )

    if baseline is None or gate.max_regression is None:
        return GateResult(
            gate=gate,
            passed=True,
            observed=observed,
            threshold=gate.max_value,
            detail=f"{observed:.4g}"
            + (f" (ceiling {gate.max_value:.4g})" if gate.max_value is not None else ""),
        )

    base_value = _observed(baseline, gate.metric, gate.tag)
    allowed = base_value * gate.max_regression if gate.relative else gate.max_regression
    delta = observed - base_value
    within_budget = delta <= allowed

    comparison: Comparison | None = None
    if gate.require_significance and gate.metric in (Metric.CER, Metric.WER):
        base_samples, cand_samples = paired_samples(
            baseline, candidate, metric=gate.metric, tag=gate.tag
        )
        if len(base_samples) >= 2:
            comparison = paired_bootstrap(base_samples, cand_samples, seed=seed)

    if within_budget:
        return GateResult(
            gate=gate,
            passed=True,
            observed=observed,
            baseline=base_value,
            threshold=allowed,
            comparison=comparison,
            detail=f"{base_value:.4g} -> {observed:.4g} (d={delta:+.4g}, budget {allowed:+.4g})",
        )

    # Over budget. Without significance the difference is indistinguishable
    # from run-to-run noise, so it warns rather than blocking a merge.
    if gate.require_significance and comparison is not None and not comparison.significant:
        return GateResult(
            gate=gate,
            passed=True,
            warning=True,
            observed=observed,
            baseline=base_value,
            threshold=allowed,
            comparison=comparison,
            detail=(
                f"{base_value:.4g} -> {observed:.4g} (d={delta:+.4g}) over budget "
                f"but not significant (p={comparison.p_value:.3f}, "
                f"CI {comparison.delta_interval.lower:+.4g}..{comparison.delta_interval.upper:+.4g})"
            ),
        )

    significance = f", p={comparison.p_value:.3f}" if comparison is not None else ""
    return GateResult(
        gate=gate,
        passed=False,
        observed=observed,
        baseline=base_value,
        threshold=allowed,
        comparison=comparison,
        detail=(
            f"{base_value:.4g} -> {observed:.4g} (d={delta:+.4g}) "
            f"exceeds budget {allowed:+.4g}{significance}"
        ),
    )


def check_regression(
    candidate: EvalReport,
    baseline: EvalReport | None,
    gates: Sequence[Gate],
    *,
    seed: int = 0,
) -> RegressionReport:
    """Evaluate every gate and collect the outcomes."""
    return RegressionReport(
        system=candidate.system,
        baseline_system=baseline.system if baseline else "(no baseline)",
        results=[evaluate_gate(gate, candidate, baseline, seed=seed) for gate in gates],
    )


#: A sensible starting policy: hold the line on quality, cost and reliability,
#: and treat Japanese as its own gate so an English-driven improvement cannot
#: mask a Japanese regression by averaging over it.
DEFAULT_GATES: tuple[Gate, ...] = (
    Gate(metric=Metric.CER, max_regression=0.05, relative=True),
    Gate(metric=Metric.FAILURE_RATE, max_value=0.02),
    Gate(metric=Metric.COST_PER_AUDIO_HOUR, max_regression=0.20, relative=True),
    Gate(metric=Metric.LATENCY_P95, max_regression=0.25, relative=True),
)
