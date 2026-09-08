"""Error metrics: WER, CER, alignment, and DER."""

from __future__ import annotations

import pytest

from koe.domain.transcript import Diarization, SpeakerTurn
from koe.evaluation.metrics import (
    Unit,
    align,
    character_error_rate,
    diarization_error_rate,
    score_transcription,
    speaker_count_error,
    word_error_rate,
)
from koe.text.script import Language

# --------------------------------------------------------------------------
# word error rate
# --------------------------------------------------------------------------


def test_perfect_transcription_scores_zero() -> None:
    rate = word_error_rate("the cat sat on the mat", "the cat sat on the mat")
    assert rate.value == 0.0
    assert rate.errors == 0


def test_single_substitution() -> None:
    rate = word_error_rate("the cat sat", "the cat sit")
    assert rate.substitutions == 1
    assert rate.value == pytest.approx(1 / 3)


def test_deletion_and_insertion_are_counted_separately() -> None:
    deleted = word_error_rate("the cat sat down", "the cat sat")
    assert (deleted.deletions, deleted.substitutions, deleted.insertions) == (1, 0, 0)

    inserted = word_error_rate("the cat sat", "the cat sat down")
    assert (inserted.insertions, inserted.substitutions, inserted.deletions) == (1, 0, 0)


def test_error_rate_is_not_clamped_at_one() -> None:
    """A hallucinated paragraph on a short utterance really is >100% error."""
    rate = word_error_rate("yes", "yes and furthermore i would like to add several things")
    assert rate.value > 1.0


def test_empty_reference_with_output_reports_the_insertions() -> None:
    """Text invented over silence is the failure mode this must not hide."""
    rate = word_error_rate("", "hello there")
    assert rate.value == 2.0


def test_empty_both_is_perfect() -> None:
    assert word_error_rate("", "").value == 0.0


# --------------------------------------------------------------------------
# character error rate / Japanese
# --------------------------------------------------------------------------


def test_japanese_cer() -> None:
    rate = character_error_rate("本日は会議です", "本日は会議でした")
    assert rate.unit is Unit.CHARACTER
    assert rate.errors > 0


def test_cer_ignores_formatting_differences() -> None:
    """Normalization is applied to both sides before comparison."""
    rate = character_error_rate("本日は、KPIを確認します。", "本日は ＫＰＩ を 確認します")
    assert rate.value == 0.0


def test_numeral_spelling_is_not_an_error() -> None:
    assert character_error_rate("二千二十五年", "2025年").value == 0.0


def test_score_transcription_picks_cer_for_japanese() -> None:
    """Japanese WER depends on the tokenizer; CER does not, so CER leads."""
    score = score_transcription("本日は会議です", "本日は会議でした")
    assert score.language is Language.JA
    assert score.primary is score.cer


def test_score_transcription_picks_wer_for_english() -> None:
    score = score_transcription("the cat sat", "the cat sit")
    assert score.language is Language.EN
    assert score.primary is score.wer


def test_wer_records_its_tokenizer() -> None:
    """A Japanese WER without its segmenter's name is not comparable."""
    assert word_error_rate("本日は会議です", "本日は会議").tokenizer != ""
    assert word_error_rate("the cat", "the cat").tokenizer == "whitespace"


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------


def test_pooling_weights_by_reference_length() -> None:
    """Corpus rate is the pooled total, not the mean of per-utterance rates."""
    short = word_error_rate("a b", "a x")  # 1/2
    long = word_error_rate(" ".join("abcdefgh"), " ".join("abcdefgh"))  # 0/8

    pooled = short + long

    assert pooled.reference_length == 10
    assert pooled.value == pytest.approx(0.1)
    # the naive mean would have been 0.25
    assert pooled.value != pytest.approx(0.25)


def test_pooling_across_units_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot pool"):
        _ = character_error_rate("a", "a") + word_error_rate("a", "a")


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_alignment_labels_each_operation() -> None:
    ops = align(["the", "cat", "sat"], ["the", "dog", "sat"])
    assert [o.op for o in ops] == ["equal", "sub", "equal"]
    assert ops[1].reference == "cat"
    assert ops[1].hypothesis == "dog"


def test_alignment_covers_insertions_and_deletions() -> None:
    ops = align(["a", "b"], ["a", "x", "b"])
    assert [o.op for o in ops].count("ins") == 1


# --------------------------------------------------------------------------
# diarization error rate
# --------------------------------------------------------------------------


def diar(*turns: tuple[str, float, float]) -> Diarization:
    return Diarization(turns=[SpeakerTurn(speaker=s, start=a, end=b) for s, a, b in turns])


def test_identical_diarization_scores_zero() -> None:
    reference = diar(("A", 0, 10), ("B", 10, 20))
    assert diarization_error_rate(reference, reference).value == pytest.approx(0.0)


def test_der_is_invariant_to_label_permutation() -> None:
    """Diarization labels are arbitrary; speaker_0 carries no meaning."""
    reference = diar(("田中", 0, 10), ("佐藤", 10, 20))
    hypothesis = diar(("speaker_1", 0, 10), ("speaker_0", 10, 20))

    score = diarization_error_rate(reference, hypothesis)

    assert score.value == pytest.approx(0.0)
    assert score.mapping["speaker_1"] == "田中"


def test_missed_speech_is_counted() -> None:
    reference = diar(("A", 0, 10))
    hypothesis = diar(("A", 0, 5))

    score = diarization_error_rate(reference, hypothesis)

    assert score.missed == pytest.approx(5.0, abs=0.05)
    assert score.value == pytest.approx(0.5, abs=0.01)


def test_false_alarm_is_counted() -> None:
    reference = diar(("A", 0, 5))
    hypothesis = diar(("A", 0, 10))

    score = diarization_error_rate(reference, hypothesis)

    assert score.false_alarm == pytest.approx(5.0, abs=0.05)


def test_speaker_confusion_is_reported_separately() -> None:
    """Confusion attributes a real sentence to the wrong person."""
    reference = diar(("A", 0, 10), ("B", 10, 20))
    hypothesis = diar(("X", 0, 20))  # one speaker where there were two

    score = diarization_error_rate(reference, hypothesis)

    assert score.confusion > 0
    assert score.missed == pytest.approx(0.0, abs=0.05)


def test_collar_forgives_boundary_disagreement() -> None:
    """Annotators disagree on boundaries by more than systems do."""
    reference = diar(("A", 0, 10), ("B", 10, 20))
    hypothesis = diar(("A", 0, 10.2), ("B", 10.2, 20))

    strict = diarization_error_rate(reference, hypothesis, collar=0.0)
    forgiving = diarization_error_rate(reference, hypothesis, collar=0.25)

    assert strict.value > forgiving.value
    assert forgiving.value == pytest.approx(0.0, abs=1e-9)


def test_speaker_count_error_is_signed() -> None:
    reference = diar(("A", 0, 5), ("B", 5, 10), ("C", 10, 15))
    assert speaker_count_error(reference, diar(("X", 0, 15))) == -2
    assert speaker_count_error(reference, reference) == 0


def test_empty_reference_diarization() -> None:
    assert diarization_error_rate(Diarization(), Diarization()).value == 0.0
