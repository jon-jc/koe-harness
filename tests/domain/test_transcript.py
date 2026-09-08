"""Transcript types and ASR x diarization fusion.

Fusion is where the two models' disagreements turn into product bugs, so these
tests are mostly about boundaries and the cases where the honest answer is
"unknown".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from koe.domain.transcript import (
    UNKNOWN_SPEAKER,
    Diarization,
    Segment,
    SpeakerTurn,
    Transcript,
    Word,
    attribute_speakers,
)
from koe.text.script import Language


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_a_span_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValidationError):
        Word(text="x", start=5.0, end=1.0)
    with pytest.raises(ValidationError):
        Segment(text="x", start=5.0, end=1.0)


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Word(text="x", start=0, end=1, confidence=1.5)


# --------------------------------------------------------------------------
# transcript assembly
# --------------------------------------------------------------------------


def test_japanese_text_joins_without_spaces() -> None:
    """Inserting spaces into Japanese corrupts it for display and comparison."""
    transcript = Transcript(
        segments=[
            Segment(text="本日の議題は", start=0, end=1, language=Language.JA),
            Segment(text="売上レビューです", start=1, end=2, language=Language.JA),
        ],
        language=Language.JA,
    )
    assert transcript.text == "本日の議題は売上レビューです"


def test_english_text_joins_with_spaces() -> None:
    transcript = Transcript(
        segments=[
            Segment(text="today's agenda", start=0, end=1, language=Language.EN),
            Segment(text="is the review", start=1, end=2, language=Language.EN),
        ],
        language=Language.EN,
    )
    assert transcript.text == "today's agenda is the review"


def test_partial_segments_are_excluded_from_text() -> None:
    """Provisional text must never reach a published document."""
    transcript = Transcript(
        segments=[
            Segment(text="確定", start=0, end=1, language=Language.JA, is_final=True),
            Segment(text="途中", start=1, end=2, language=Language.JA, is_final=False),
        ],
        language=Language.JA,
    )
    assert transcript.text == "確定"
    assert len(transcript.final_segments) == 1


def test_dominant_language_is_duration_weighted() -> None:
    transcript = Transcript(
        segments=[
            Segment(text="hello", start=0, end=1, language=Language.EN),
            Segment(text="本日の議題", start=1, end=20, language=Language.JA),
        ]
    )
    assert transcript.dominant_language() is Language.JA


def test_transcript_lines_render_speaker_prefixes() -> None:
    transcript = Transcript(
        segments=[
            Segment(text="始めます", start=0, end=1, speaker="田中"),
            Segment(text="はい", start=1, end=2, speaker="佐藤"),
        ]
    )
    assert transcript.transcript_lines() == ["田中: 始めます", "佐藤: はい"]


# --------------------------------------------------------------------------
# diarization
# --------------------------------------------------------------------------


def test_speaking_time_totals_per_speaker() -> None:
    diarization = Diarization(
        turns=[
            SpeakerTurn(speaker="A", start=0, end=10),
            SpeakerTurn(speaker="B", start=10, end=13),
            SpeakerTurn(speaker="A", start=13, end=15),
        ]
    )
    assert diarization.speaking_time() == {"A": 12.0, "B": 3.0}
    assert diarization.num_speakers == 2


def test_overlap_and_distance() -> None:
    turn = SpeakerTurn(speaker="A", start=10, end=20)
    assert turn.overlap(15, 25) == 5.0
    assert turn.overlap(0, 5) == 0.0
    assert turn.distance_to(15, 25) == 0.0
    assert turn.distance_to(22, 25) == 2.0
    assert turn.distance_to(5, 8) == 2.0


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def test_words_are_attributed_by_maximum_overlap() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                text="ab",
                start=0,
                end=4,
                language=Language.EN,
                words=[word("a", 0, 2), word("b", 2, 4)],
            )
        ]
    )
    diarization = Diarization(
        turns=[
            SpeakerTurn(speaker="田中", start=0, end=2),
            SpeakerTurn(speaker="佐藤", start=2, end=4),
        ]
    )

    result = attribute_speakers(transcript, diarization)

    assert [s.speaker for s in result.segments] == ["田中", "佐藤"]
    assert [s.text for s in result.segments] == ["a", "b"]


def test_a_segment_spanning_a_speaker_change_is_split() -> None:
    """An interruption mid-sentence must not be collapsed onto one speaker."""
    transcript = Transcript(
        segments=[
            Segment(
                text="one two three four",
                start=0,
                end=4,
                language=Language.EN,
                words=[
                    word("one", 0, 1),
                    word("two", 1, 2),
                    word("three", 2, 3),
                    word("four", 3, 4),
                ],
            )
        ]
    )
    diarization = Diarization(
        turns=[
            SpeakerTurn(speaker="Alice", start=0, end=2),
            SpeakerTurn(speaker="Bob", start=2, end=4),
        ]
    )

    result = attribute_speakers(transcript, diarization)

    assert len(result.segments) == 2
    assert result.segments[0].speaker == "Alice"
    assert result.segments[0].text == "one two"
    assert result.segments[1].speaker == "Bob"
    assert result.segments[1].text == "three four"


def test_japanese_split_segments_rejoin_without_spaces() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                text="はいそうです",
                start=0,
                end=6,
                language=Language.JA,
                words=[word(ch, i, i + 1) for i, ch in enumerate("はいそうです")],
            )
        ]
    )
    diarization = Diarization(
        turns=[
            SpeakerTurn(speaker="A", start=0, end=2),
            SpeakerTurn(speaker="B", start=2, end=6),
        ]
    )

    result = attribute_speakers(transcript, diarization)

    assert [s.text for s in result.segments] == ["はい", "そうです"]


def test_a_word_far_from_every_turn_is_unknown_rather_than_guessed() -> None:
    """Confidently attributing a commitment to the wrong person is the worst outcome."""
    transcript = Transcript(
        segments=[
            Segment(
                text="orphan",
                start=100,
                end=101,
                language=Language.EN,
                words=[word("orphan", 100, 101)],
            )
        ]
    )
    diarization = Diarization(turns=[SpeakerTurn(speaker="Alice", start=0, end=2)])

    result = attribute_speakers(transcript, diarization, max_gap=0.5)

    assert result.segments[0].speaker == UNKNOWN_SPEAKER


def test_a_word_just_past_a_boundary_snaps_to_the_nearest_turn() -> None:
    """The two models routinely disagree by a few hundred milliseconds."""
    transcript = Transcript(
        segments=[
            Segment(
                text="edge",
                start=2.1,
                end=2.4,
                language=Language.EN,
                words=[word("edge", 2.1, 2.4)],
            )
        ]
    )
    diarization = Diarization(turns=[SpeakerTurn(speaker="Alice", start=0, end=2)])

    result = attribute_speakers(transcript, diarization, max_gap=0.5)

    assert result.segments[0].speaker == "Alice"


def test_segment_level_fallback_when_the_asr_gives_no_word_timings() -> None:
    transcript = Transcript(
        segments=[Segment(text="no words here", start=0, end=2, language=Language.EN)]
    )
    diarization = Diarization(turns=[SpeakerTurn(speaker="Bob", start=0, end=2)])

    result = attribute_speakers(transcript, diarization)

    assert result.segments[0].speaker == "Bob"
    assert len(result.segments) == 1


def test_fusion_without_diarization_is_a_no_op() -> None:
    transcript = Transcript(segments=[Segment(text="x", start=0, end=1)])
    assert attribute_speakers(transcript, Diarization()) is transcript


def test_by_speaker_groups_final_segments() -> None:
    transcript = Transcript(
        segments=[
            Segment(text="a", start=0, end=1, speaker="A"),
            Segment(text="b", start=1, end=2, speaker="B"),
            Segment(text="c", start=2, end=3, speaker="A"),
        ]
    )
    grouped = transcript.by_speaker()
    assert set(grouped) == {"A", "B"}
    assert len(grouped["A"]) == 2
