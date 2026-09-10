"""The VAD scorer.

A benchmark is only worth the confidence placed in its metrics, and these are
the metrics a change to the detector will be judged by. So they are tested
against hand-computed cases -- including the ones where a plausible
implementation quietly reports the flattering answer.
"""

from __future__ import annotations

import pytest

from koe.evaluation.vad import Span, VADReport, score_vad


def test_a_perfect_detector_scores_perfectly() -> None:
    reference = [Span(1.0, 3.0), Span(5.0, 6.0)]
    score = score_vad(reference, list(reference), audio_seconds=10.0)

    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.detected == 2
    assert score.missed == 0
    assert score.false_alarms == 0


def test_detecting_nothing_costs_recall_and_not_precision() -> None:
    """Precision over no detections is 1.0 by convention, which is why it is
    never reported alone."""
    score = score_vad([Span(1.0, 3.0)], [], audio_seconds=10.0)

    assert score.recall == 0.0
    assert score.missed == 1
    assert score.false_alarms == 0


def test_detecting_everything_costs_precision_and_not_recall() -> None:
    score = score_vad([Span(1.0, 3.0)], [Span(0.0, 10.0)], audio_seconds=10.0)

    assert score.recall == 1.0
    assert score.precision == pytest.approx(0.2, abs=0.01)


def test_a_split_utterance_is_distinguished_from_a_clipped_one() -> None:
    """They score almost identically at frame level and are not remotely the
    same to read: a split sentence is recognized as fragments, each without the
    context of the others."""
    reference = [Span(1.0, 5.0)]
    split = score_vad(reference, [Span(1.0, 2.8), Span(3.2, 5.0)], audio_seconds=10.0)
    clipped = score_vad(reference, [Span(1.0, 4.6)], audio_seconds=10.0)

    assert split.split == 1
    assert clipped.split == 0
    assert split.f1 == pytest.approx(clipped.f1, abs=0.05)


def test_two_utterances_merged_into_one_are_counted_as_merged() -> None:
    """The failure the unvoiced-run rule exists to prevent."""
    score = score_vad(
        [Span(1.0, 3.0), Span(4.0, 6.0)],
        [Span(1.0, 6.0)],
        audio_seconds=10.0,
    )
    assert score.merged == 1
    assert score.detected == 2
    assert score.false_alarms == 0


def test_a_detection_over_nothing_is_a_false_alarm() -> None:
    score = score_vad([Span(1.0, 3.0)], [Span(1.0, 3.0), Span(7.0, 8.0)], audio_seconds=10.0)
    assert score.false_alarms == 1
    assert score.detected == 1


def test_false_alarms_are_reported_per_minute_of_audio() -> None:
    """Not as a rate over detections, which lets a detector that fires
    constantly look good by also catching a lot of real speech."""
    score = score_vad([], [Span(1.0, 2.0), Span(3.0, 4.0)], audio_seconds=30.0)
    assert score.false_alarms_per_minute == pytest.approx(4.0)


def test_boundary_error_is_signed_so_late_is_distinguishable_from_imprecise() -> None:
    """A detector that is consistently late and one that is merely noisy have
    the same mean absolute error and different fixes."""
    late = score_vad([Span(1.0, 3.0)], [Span(1.2, 3.2)], audio_seconds=10.0)
    assert late.onset_bias_ms == pytest.approx(200.0, abs=1.0)
    assert late.offset_bias_ms == pytest.approx(200.0, abs=1.0)

    noisy = score_vad(
        [Span(1.0, 3.0), Span(5.0, 7.0)],
        [Span(1.2, 3.0), Span(4.8, 7.0)],
        audio_seconds=10.0,
    )
    assert noisy.onset_bias_ms == pytest.approx(0.0, abs=1.0)
    assert noisy.onset_mae_ms == pytest.approx(200.0, abs=1.0)


def test_a_barely_touching_detection_does_not_count_as_the_utterance() -> None:
    """Otherwise a detector that clips 90% of every sentence reports finding
    all of them."""
    score = score_vad([Span(1.0, 5.0)], [Span(4.8, 5.2)], audio_seconds=10.0)
    assert score.detected == 0
    assert score.missed == 1
    assert score.false_alarms == 1


def test_conditions_are_reported_separately_as_well_as_pooled() -> None:
    """An average over a quiet room and a noisy one hides exactly the case
    that motivated the change."""
    report = VADReport()
    report.add("quiet", score_vad([Span(1.0, 3.0)], [Span(1.0, 3.0)], audio_seconds=10.0))
    report.add("noisy", score_vad([Span(1.0, 3.0)], [], audio_seconds=10.0))

    assert report.scores["quiet"].recall == 1.0
    assert report.scores["noisy"].recall == 0.0
    assert report.total.recall == pytest.approx(0.5, abs=0.01)
    assert report.total.audio_seconds == 20.0


def test_pooling_concatenates_boundary_errors_rather_than_averaging_averages() -> None:
    """Averaging per-condition means would weight a condition with one
    utterance the same as one with fifty."""
    report = VADReport()
    report.add("a", score_vad([Span(1.0, 3.0)], [Span(1.1, 3.0)], audio_seconds=10.0))
    report.add(
        "b",
        score_vad(
            [Span(1.0, 3.0), Span(5.0, 7.0)],
            [Span(1.3, 3.0), Span(5.3, 7.0)],
            audio_seconds=10.0,
        ),
    )
    assert len(report.total.onset_errors_ms) == 3
    assert report.total.onset_bias_ms == pytest.approx((100 + 300 + 300) / 3, abs=2.0)


def test_the_report_serializes_for_a_regression_gate() -> None:
    report = VADReport()
    report.add("quiet", score_vad([Span(1.0, 3.0)], [Span(1.0, 3.0)], audio_seconds=10.0))
    payload = report.to_dict()

    assert payload["conditions"]["quiet"]["f1"] == 1.0
    assert payload["total"]["audio_seconds"] == 10.0
