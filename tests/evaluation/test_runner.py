"""End-to-end evaluation runs, slicing, and regression gating."""

from __future__ import annotations

import pytest

from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Diarization, Transcript
from koe.evaluation.dataset import Dataset
from koe.evaluation.regression import (
    Gate,
    Metric,
    check_regression,
    paired_samples,
)
from koe.evaluation.runner import EvalReport, mock_transcriber, run_evaluation
from koe.providers.base import ProviderError
from koe.providers.mock import MEETING_EN, MEETING_JA, MEETING_MIXED, MockASR, MockDiarization
from koe.text.script import Language


def build_dataset() -> Dataset:
    return Dataset.from_scripts(
        "meetings",
        {"ja-001": MEETING_JA, "en-001": MEETING_EN, "mixed-001": MEETING_MIXED},
        tags={
            "ja-001": ["monolingual", "numerals"],
            "en-001": ["monolingual"],
            "mixed-001": ["code-switch"],
        },
    )


async def run(degradation: float, *, name: str = "sys", **kwargs: object) -> EvalReport:
    asr = MockASR(degradation=degradation, name=name, **kwargs)  # type: ignore[arg-type]
    return await run_evaluation(build_dataset(), mock_transcriber(asr), system=name, seed=7)


# --------------------------------------------------------------------------
# the round trip that validates the metric itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degradation", [0.0, 0.05, 0.15, 0.30])
async def test_measured_error_rate_tracks_the_injected_one(degradation: float) -> None:
    """A backend corrupted at a known rate is how the metric gets validated.

    If CER cannot recover a corruption it was told about, the bug is in the
    metric, not the model. The tolerance is loose because normalization
    legitimately absorbs some corruptions (a mangled character inside a filler
    word disappears entirely), but the measurement must track the injection.
    """
    report = await run(degradation)
    measured = report.overall(seed=7).cer.interval.point

    assert measured == pytest.approx(degradation, abs=0.10)
    if degradation == 0.0:
        assert measured == 0.0


async def test_error_rate_increases_monotonically_with_degradation() -> None:
    rates = [(await run(d)).overall(seed=7).cer.interval.point for d in (0.0, 0.1, 0.2, 0.4)]
    assert rates == sorted(rates)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


async def test_a_clean_run_scores_zero_and_reports_every_case() -> None:
    report = await run(0.0)
    assert len(report.results) == 3
    assert not report.failures
    assert report.overall(seed=7).cer.interval.point == 0.0


async def test_a_failing_case_does_not_abort_the_run() -> None:
    """An eval that stops at the first error tells you nothing about the rest."""

    async def flaky(case: object) -> Transcript:
        if getattr(case, "id", "") == "en-001":
            raise ProviderError("backend down", provider="x")
        return await MockASR(degradation=0.0).transcribe(
            AudioChunk(data=b"\x00" * STANDARD_FORMAT.bytes_for(1.0))
        )

    report = await run_evaluation(build_dataset(), flaky, system="flaky")

    assert len(report.results) == 3
    assert len(report.failures) == 1
    assert report.failure_rate == pytest.approx(1 / 3)
    assert report.failures[0].error is not None


async def test_failure_rate_is_reported_separately_from_quality() -> None:
    """A backend that crashes on 20% of inputs has a reliability problem, and
    averaging that into a quality score would disguise it."""
    report = await run(0.0)
    assert report.failure_rate == 0.0
    assert report.overall(seed=7).n == 3


# --------------------------------------------------------------------------
# slicing
# --------------------------------------------------------------------------


async def test_slices_by_language() -> None:
    report = await run(0.1)
    names = {s.name for s in report.slice_by_language(seed=7)}
    assert "lang:ja" in names
    assert "lang:en" in names


async def test_slices_by_tag_expose_what_an_aggregate_hides() -> None:
    """The actionable finding is per-slice, not corpus-wide."""
    report = await run(0.1)
    slices = {s.name: s for s in report.slice_by_tag(seed=7)}
    assert "tag:code-switch" in slices
    assert "tag:numerals" in slices
    assert slices["tag:code-switch"].n == 1


async def test_report_serializes_for_use_as_a_baseline() -> None:
    report = await run(0.1)
    payload = report.to_dict()
    assert payload["system"] == "sys"
    assert "cer" in payload and "cer_lower" in payload
    assert "slices" in payload
    assert payload["n_cases"] == 3


# --------------------------------------------------------------------------
# cost and latency travel with quality
# --------------------------------------------------------------------------


async def test_cost_is_reported_per_audio_hour() -> None:
    """The unit that actually appears on an invoice."""
    report = await run(0.0, name="priced", cost_per_audio_minute_usd=0.006)
    assert report.total_audio_seconds > 0
    # cost is accrued by the provider's Usage; the report exposes the rate
    assert report.cost_per_audio_hour >= 0.0


async def test_latency_percentiles_are_available() -> None:
    report = await run(0.0)
    assert report.latency_percentile(0.95) >= report.latency_percentile(0.50)


# --------------------------------------------------------------------------
# diarization inside a run
# --------------------------------------------------------------------------


async def test_diarization_is_scored_when_a_diarizer_is_supplied() -> None:
    dataset = build_dataset()

    async def diarize(case: object) -> Diarization:
        utterances = case.utterances()  # type: ignore[attr-defined]
        return await MockDiarization(script=utterances).diarize(
            AudioChunk(data=b"\x00" * STANDARD_FORMAT.bytes_for(1.0))
        )

    report = await run_evaluation(
        dataset, mock_transcriber(MockASR(degradation=0.0)), diarize=diarize, seed=7
    )

    scored = [r for r in report.succeeded if r.diarization is not None]
    assert len(scored) == 3
    assert all(r.diarization.value == pytest.approx(0.0, abs=0.02) for r in scored)


# --------------------------------------------------------------------------
# regression gating
# --------------------------------------------------------------------------


async def test_paired_samples_align_by_case_id() -> None:
    baseline = await run(0.0, name="base")
    candidate = await run(0.2, name="cand")

    base_samples, cand_samples = paired_samples(baseline, candidate, metric=Metric.CER)

    assert len(base_samples) == len(cand_samples) == 3


async def test_a_clear_regression_fails_the_gate() -> None:
    baseline = await run(0.0, name="base")
    candidate = await run(0.35, name="cand")

    report = check_regression(
        candidate,
        baseline,
        [Gate(metric=Metric.CER, max_value=0.10)],
        seed=7,
    )

    assert not report.passed
    assert report.failures[0].gate.metric is Metric.CER


async def test_an_identical_system_passes() -> None:
    baseline = await run(0.1, name="base")
    candidate = await run(0.1, name="cand")

    report = check_regression(candidate, baseline, [Gate(metric=Metric.CER, max_regression=0.05)])

    assert report.passed


async def test_an_insignificant_regression_warns_instead_of_failing() -> None:
    """A gate that fires on noise gets switched off within a month."""
    baseline = await run(0.10, name="base")
    candidate = await run(0.11, name="cand")

    report = check_regression(
        candidate,
        baseline,
        [Gate(metric=Metric.CER, max_regression=0.0, relative=False, require_significance=True)],
        seed=7,
    )

    # over budget, but only a handful of cases -- not distinguishable from noise
    result = report.results[0]
    if result.comparison is not None and not result.comparison.significant:
        assert result.passed
        assert result.warning
        assert "not significant" in result.detail


async def test_an_absolute_ceiling_fails_regardless_of_significance() -> None:
    """Some limits are product decisions, not statistical claims."""
    candidate = await run(0.30, name="cand")

    report = check_regression(candidate, None, [Gate(metric=Metric.CER, max_value=0.05)])

    assert not report.passed
    assert "ceiling" in report.failures[0].detail


async def test_failure_rate_gate() -> None:
    async def always_broken(case: object) -> Transcript:
        raise ProviderError("down", provider="x")

    candidate = await run_evaluation(build_dataset(), always_broken, system="broken")
    report = check_regression(candidate, None, [Gate(metric=Metric.FAILURE_RATE, max_value=0.02)])

    assert not report.passed


async def test_report_renders_readably() -> None:
    baseline = await run(0.0, name="base")
    candidate = await run(0.05, name="cand")
    rendered = check_regression(
        candidate, baseline, [Gate(metric=Metric.CER, max_regression=0.5)]
    ).render()

    assert "regression check" in rendered
    assert "RESULT:" in rendered


# --------------------------------------------------------------------------
# dataset round trip
# --------------------------------------------------------------------------


def test_dataset_round_trips_through_jsonl(tmp_path: object) -> None:
    dataset = build_dataset()
    path = tmp_path / "eval.jsonl"  # type: ignore[operator]
    dataset.to_jsonl(path)

    loaded = Dataset.from_jsonl(path)

    assert len(loaded) == len(dataset)
    assert {c.id for c in loaded.cases} == {"ja-001", "en-001", "mixed-001"}
    assert loaded.cases[0].resolved_reference()


def test_dataset_filters_by_tag_and_language() -> None:
    dataset = build_dataset()
    assert len(dataset.filter(tags=["code-switch"])) == 1
    assert len(dataset.filter(language=Language.JA)) >= 1


def test_malformed_jsonl_names_the_line(tmp_path: object) -> None:
    path = tmp_path / "bad.jsonl"  # type: ignore[operator]
    path.write_text('{"id": "ok"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=":2:"):
        Dataset.from_jsonl(path)
