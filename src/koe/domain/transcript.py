"""Transcript and diarization types, and the fusion between them.

ASR and diarization are separate models that answer separate questions -- *what
was said* and *who was talking* -- on independent timelines. Neither output is
useful for meeting minutes on its own: "the deadline moved to Friday" only
becomes actionable once you know which participant said it.

Joining them is this module's real job, and it is where the errors live. The
two models disagree at boundaries, diarization emits turns that straddle
sentence ends, and an ASR word occasionally lands in a gap where no speaker was
detected at all. :func:`attribute_speakers` resolves those cases explicitly
rather than letting them become silent misattributions -- which are the worst
failure mode in a meeting product, because a confident transcript that assigns
a commitment to the wrong person is more damaging than one that admits it does
not know.

Unlike :mod:`koe.domain.audio`, these are pydantic models: they cross the API
boundary, get persisted, and are produced at utterance rate rather than frame
rate, so validation is cheap relative to their lifetime.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from koe.text.script import Language, primary_language

#: Speaker label used when diarization could not attribute a span.
UNKNOWN_SPEAKER = "unknown"

Timestamp = Annotated[float, Field(ge=0.0, description="seconds from session start")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Word(BaseModel):
    """A single recognized word with its timing."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: Timestamp
    end: Timestamp
    confidence: Confidence = 1.0
    speaker: str | None = None

    @model_validator(mode="after")
    def _check_span(self) -> Word:
        if self.end < self.start:
            raise ValueError(
                f"word {self.text!r} ends ({self.end}) before it starts ({self.start})"
            )
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class Segment(BaseModel):
    """A contiguous span of speech from one speaker.

    `is_final` distinguishes a stabilized result from a partial hypothesis that
    the ASR may still revise. The UI renders partials differently and the
    minutes stage ignores them entirely, so conflating the two would let
    provisional text reach a published document.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start: Timestamp
    end: Timestamp
    words: list[Word] = Field(default_factory=list)
    language: Language = Language.UNKNOWN
    confidence: Confidence = 1.0
    speaker: str | None = None
    is_final: bool = True

    @model_validator(mode="after")
    def _check_span(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"segment ends ({self.end}) before it starts ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start

    def detect_language(self) -> Language:
        """Language of this segment, inferred from script when not set."""
        if self.language is not Language.UNKNOWN:
            return self.language
        return primary_language(self.text)


class SpeakerTurn(BaseModel):
    """A span during which one speaker held the floor."""

    model_config = ConfigDict(frozen=True)

    speaker: str
    start: Timestamp
    end: Timestamp
    confidence: Confidence = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, start: float, end: float) -> float:
        """Seconds of overlap between this turn and ``[start, end]``."""
        return max(0.0, min(self.end, end) - max(self.start, start))

    def distance_to(self, start: float, end: float) -> float:
        """Gap in seconds to ``[start, end]``; 0 when they overlap."""
        if self.overlap(start, end) > 0:
            return 0.0
        return start - self.end if start > self.end else self.start - end


class Diarization(BaseModel):
    """Who spoke when."""

    turns: list[SpeakerTurn] = Field(default_factory=list)
    provider: str = ""
    model: str = ""

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def speaking_time(self) -> dict[str, float]:
        """Total seconds held by each speaker.

        Surfaced in the product as participation balance, and used internally
        as a sanity check: a meeting where one label holds 99% of the time
        usually means diarization collapsed rather than that one person talked.
        """
        totals: dict[str, float] = defaultdict(float)
        for turn in self.turns:
            totals[turn.speaker] += turn.duration
        return dict(totals)

    def at(self, start: float, end: float) -> SpeakerTurn | None:
        """Turn with the greatest overlap of ``[start, end]``."""
        best: SpeakerTurn | None = None
        best_overlap = 0.0
        for turn in self.turns:
            amount = turn.overlap(start, end)
            if amount > best_overlap:
                best, best_overlap = turn, amount
        return best


class Transcript(BaseModel):
    """The full recognition result for a session or file."""

    segments: list[Segment] = Field(default_factory=list)
    language: Language = Language.UNKNOWN
    duration: float = 0.0
    provider: str = ""
    model: str = ""

    @property
    def text(self) -> str:
        """Concatenated final text.

        Joined with spaces for English and without for Japanese, because
        inserting spaces into Japanese would corrupt the text for both display
        and any downstream comparison.
        """
        finals = [s.text for s in self.segments if s.is_final and s.text]
        if not finals:
            return ""
        joiner = "" if self.dominant_language() is Language.JA else " "
        return joiner.join(finals)

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def final_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_final]

    def dominant_language(self) -> Language:
        """Most common segment language, weighted by duration."""
        if self.language is not Language.UNKNOWN:
            return self.language
        weights: dict[Language, float] = defaultdict(float)
        for seg in self.segments:
            weights[seg.detect_language()] += max(seg.duration, 0.001)
        if not weights:
            return Language.UNKNOWN
        return max(weights.items(), key=lambda kv: kv[1])[0]

    def by_speaker(self) -> dict[str, list[Segment]]:
        grouped: dict[str, list[Segment]] = defaultdict(list)
        for seg in self.final_segments:
            grouped[seg.speaker or UNKNOWN_SPEAKER].append(seg)
        return dict(grouped)

    def transcript_lines(self) -> list[str]:
        """``speaker: text`` lines, the form the LLM stage consumes."""
        return [
            f"{seg.speaker or UNKNOWN_SPEAKER}: {seg.text}"
            for seg in self.final_segments
            if seg.text
        ]


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def _speaker_for_span(
    diarization: Diarization,
    start: float,
    end: float,
    *,
    max_gap: float,
) -> str | None:
    """Best speaker for ``[start, end]``, or ``None`` if nothing is close enough."""
    turn = diarization.at(start, end)
    if turn is not None:
        return turn.speaker

    # No overlap: the models disagree at a boundary. Attach to the nearest turn
    # only if it is close enough that adjacency is real evidence; beyond that,
    # returning None (-> "unknown") is the honest answer. Guessing here is how a
    # meeting product confidently attributes a commitment to the wrong person.
    nearest: SpeakerTurn | None = None
    nearest_distance = float("inf")
    for candidate in diarization.turns:
        distance = candidate.distance_to(start, end)
        if distance < nearest_distance:
            nearest, nearest_distance = candidate, distance
    if nearest is not None and nearest_distance <= max_gap:
        return nearest.speaker
    return None


def attribute_speakers(
    transcript: Transcript,
    diarization: Diarization,
    *,
    max_gap: float = 0.5,
) -> Transcript:
    """Join a transcript with diarization, splitting segments on speaker change.

    Word timings are used when the ASR provides them, because a single ASR
    segment routinely spans a speaker change -- one participant interrupting
    another mid-sentence -- and attributing the whole segment to one of them
    loses the interruption entirely. Segments are therefore re-cut at every
    speaker boundary.

    When word timings are absent (some backends return segment-level output
    only), attribution falls back to the segment's own span, which is the best
    available evidence.

    `max_gap` bounds how far a word may sit from a turn and still be attributed
    to it. Anything further becomes :data:`UNKNOWN_SPEAKER` rather than a guess.
    """
    if not diarization.turns:
        return transcript

    rebuilt: list[Segment] = []

    for segment in transcript.segments:
        if not segment.words:
            speaker = _speaker_for_span(diarization, segment.start, segment.end, max_gap=max_gap)
            rebuilt.append(segment.model_copy(update={"speaker": speaker or UNKNOWN_SPEAKER}))
            continue

        attributed = [
            word.model_copy(
                update={
                    "speaker": _speaker_for_span(diarization, word.start, word.end, max_gap=max_gap)
                    or UNKNOWN_SPEAKER
                }
            )
            for word in segment.words
        ]

        # Re-cut the segment wherever the speaker changes.
        run: list[Word] = []
        for word in attributed:
            if run and word.speaker != run[-1].speaker:
                rebuilt.append(_segment_from_words(run, segment))
                run = []
            run.append(word)
        if run:
            rebuilt.append(_segment_from_words(run, segment))

    return transcript.model_copy(update={"segments": rebuilt})


def _segment_from_words(words: list[Word], template: Segment) -> Segment:
    """Build a single-speaker segment from a run of attributed words."""
    language = template.language
    joiner = "" if language is Language.JA else " "
    text = joiner.join(w.text for w in words)
    if language is Language.UNKNOWN:
        language = primary_language(text)
        if language is Language.JA:
            text = "".join(w.text for w in words)
    return Segment(
        text=text,
        start=words[0].start,
        end=words[-1].end,
        words=words,
        language=language,
        confidence=min((w.confidence for w in words), default=template.confidence),
        speaker=words[0].speaker,
        is_final=template.is_final,
    )
