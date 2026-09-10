"""Scoring a voice activity detector against known speech.

"The detection got better" is not a claim anyone should accept without a
number, and the number has to be the right one. A detector is easy to improve
on one axis by quietly ruining another: raise the threshold and false alarms
vanish along with every quiet speaker, lower it and recall looks superb while
the transcript fills with furniture.

So this reports three families at once, because they fail in different
directions and a change that helps one usually costs another.

**Frame level** -- precision and recall over 10 ms slices. The workhorse, and
the one that reflects how much *audio* is right. It is also the one that
flatters a detector on long utterances: getting a ten-second span mostly right
swamps a completely missed short one.

**Segment level** -- how many real utterances came back as one utterance.
This is what a transcript is made of, so it is what a user experiences. It
separates the three ways an utterance can be wrong that frame scoring blurs
together: **missed** entirely, **split** into fragments each of which will be
recognized without the context of the others, and **merged** with a neighbour
across a pause. A split sentence and a slightly-clipped one score almost
identically at frame level and are not remotely the same to read.

**Boundary error** -- how far the edges are out, in milliseconds, signed. Late
onsets clip the first phoneme; early offsets clip the last, which in Japanese
is where the negation and the tense live. Reported separately because they have
different consequences and different fixes.

**False alarms are counted per minute, not as a rate.** A rate divides by the
number of detections, which lets a detector that fires constantly look good by
also detecting a lot of real speech. Per minute of audio is what determines
whether the transcript is usable and what the recognition bill is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Frame scoring resolution. 10 ms is finer than any endpointer's frame, so the
#: score measures the detector rather than the grid it is scored on.
RESOLUTION_S = 0.01

#: Reference and hypothesis spans count as the same utterance when they overlap
#: by at least this much of the reference. Deliberately lenient: the question a
#: match answers is "did this utterance come back at all", and the *quality* of
#: the match is then reported as boundary error rather than folded into a
#: yes/no.
MATCH_OVERLAP = 0.30


@dataclass(frozen=True, slots=True)
class Span:
    """A span of speech, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: Span) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True, slots=True)
class VADScore:
    """What one run of a detector got right and wrong."""

    # frame level
    true_positive_frames: int = 0
    false_positive_frames: int = 0
    false_negative_frames: int = 0

    # segment level
    reference_segments: int = 0
    hypothesis_segments: int = 0
    detected: int = 0
    missed: int = 0
    false_alarms: int = 0
    split: int = 0
    merged: int = 0

    # boundaries, in milliseconds, signed: positive means late
    onset_errors_ms: tuple[float, ...] = ()
    offset_errors_ms: tuple[float, ...] = ()

    audio_seconds: float = 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positive_frames + self.false_positive_frames
        return self.true_positive_frames / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive_frames + self.false_negative_frames
        return self.true_positive_frames / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def detection_rate(self) -> float:
        """Fraction of real utterances that came back as at least one segment."""
        return self.detected / self.reference_segments if self.reference_segments else 1.0

    @property
    def false_alarms_per_minute(self) -> float:
        minutes = self.audio_seconds / 60.0
        return self.false_alarms / minutes if minutes else 0.0

    @property
    def onset_mae_ms(self) -> float:
        return _mean(abs(value) for value in self.onset_errors_ms)

    @property
    def offset_mae_ms(self) -> float:
        return _mean(abs(value) for value in self.offset_errors_ms)

    @property
    def onset_bias_ms(self) -> float:
        """Signed, so a detector that is consistently late is distinguishable
        from one that is merely imprecise. They have different fixes."""
        return _mean(iter(self.onset_errors_ms))

    @property
    def offset_bias_ms(self) -> float:
        return _mean(iter(self.offset_errors_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "detection_rate": round(self.detection_rate, 4),
            "reference_segments": self.reference_segments,
            "hypothesis_segments": self.hypothesis_segments,
            "detected": self.detected,
            "missed": self.missed,
            "false_alarms": self.false_alarms,
            "split": self.split,
            "merged": self.merged,
            "false_alarms_per_minute": round(self.false_alarms_per_minute, 3),
            "onset_mae_ms": round(self.onset_mae_ms, 1),
            "offset_mae_ms": round(self.offset_mae_ms, 1),
            "onset_bias_ms": round(self.onset_bias_ms, 1),
            "offset_bias_ms": round(self.offset_bias_ms, 1),
            "audio_seconds": round(self.audio_seconds, 2),
        }


def _mean(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _mask(spans: Sequence[Span], slices: int, resolution: float) -> bytearray:
    """Spans as a per-slice occupancy mask."""
    mask = bytearray(slices)
    for span in spans:
        first = max(0, int(span.start / resolution))
        last = min(slices, round(span.end / resolution))
        for index in range(first, last):
            mask[index] = 1
    return mask


def score_vad(
    reference: Sequence[Span],
    hypothesis: Sequence[Span],
    *,
    audio_seconds: float,
    resolution: float = RESOLUTION_S,
    match_overlap: float = MATCH_OVERLAP,
) -> VADScore:
    """Score `hypothesis` against `reference`.

    `audio_seconds` is required rather than inferred from the spans, because
    the interesting denominator for false alarms is how much audio was
    listened to -- and a detector that found nothing has no spans to infer it
    from.
    """
    slices = max(1, round(audio_seconds / resolution))
    reference_mask = _mask(reference, slices, resolution)
    hypothesis_mask = _mask(hypothesis, slices, resolution)

    true_positive = false_positive = false_negative = 0
    for index in range(slices):
        in_reference = reference_mask[index]
        in_hypothesis = hypothesis_mask[index]
        if in_reference and in_hypothesis:
            true_positive += 1
        elif in_hypothesis:
            false_positive += 1
        elif in_reference:
            false_negative += 1

    # -- segment level ----------------------------------------------------
    # Every pairing that clears the overlap bar, so one reference matching two
    # hypotheses is visible as a split rather than silently counted once.
    matches: list[tuple[int, int, float]] = []
    for r_index, r_span in enumerate(reference):
        for h_index, h_span in enumerate(hypothesis):
            shared = r_span.overlap(h_span)
            if r_span.duration > 0 and shared / r_span.duration >= match_overlap:
                matches.append((r_index, h_index, shared))

    matched_references = {r for r, _, _ in matches}
    matched_hypotheses = {h for _, h, _ in matches}

    hypotheses_per_reference: dict[int, int] = {}
    references_per_hypothesis: dict[int, int] = {}
    for r_index, h_index, _ in matches:
        hypotheses_per_reference[r_index] = hypotheses_per_reference.get(r_index, 0) + 1
        references_per_hypothesis[h_index] = references_per_hypothesis.get(h_index, 0) + 1

    split = sum(1 for count in hypotheses_per_reference.values() if count > 1)
    merged = sum(1 for count in references_per_hypothesis.values() if count > 1)

    # -- boundaries -------------------------------------------------------
    # Scored against the single best-overlapping hypothesis, since a split
    # reference has no one pair of edges to compare.
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    for r_index, r_span in enumerate(reference):
        candidates = [(shared, h) for r, h, shared in matches if r == r_index]
        if not candidates:
            continue
        _, best = max(candidates)
        h_span = hypothesis[best]
        onset_errors.append((h_span.start - r_span.start) * 1000.0)
        offset_errors.append((h_span.end - r_span.end) * 1000.0)

    return VADScore(
        true_positive_frames=true_positive,
        false_positive_frames=false_positive,
        false_negative_frames=false_negative,
        reference_segments=len(reference),
        hypothesis_segments=len(hypothesis),
        detected=len(matched_references),
        missed=len(reference) - len(matched_references),
        false_alarms=len(hypothesis) - len(matched_hypotheses),
        split=split,
        merged=merged,
        onset_errors_ms=tuple(onset_errors),
        offset_errors_ms=tuple(offset_errors),
        audio_seconds=audio_seconds,
    )


@dataclass(slots=True)
class VADReport:
    """Scores across a set of conditions, and their total.

    Conditions are kept apart rather than pooled because the whole point of a
    VAD benchmark is that a detector behaves differently in a quiet room and a
    noisy one, and an average over both hides exactly the case that motivated
    the change.
    """

    scores: dict[str, VADScore] = field(default_factory=dict)

    def add(self, condition: str, score: VADScore) -> None:
        self.scores[condition] = score

    @property
    def total(self) -> VADScore:
        """Everything pooled: frame counts add, boundary errors concatenate."""
        pooled = VADScore()
        onsets: list[float] = []
        offsets: list[float] = []
        for score in self.scores.values():
            onsets.extend(score.onset_errors_ms)
            offsets.extend(score.offset_errors_ms)
            pooled = VADScore(
                true_positive_frames=pooled.true_positive_frames + score.true_positive_frames,
                false_positive_frames=pooled.false_positive_frames + score.false_positive_frames,
                false_negative_frames=pooled.false_negative_frames + score.false_negative_frames,
                reference_segments=pooled.reference_segments + score.reference_segments,
                hypothesis_segments=pooled.hypothesis_segments + score.hypothesis_segments,
                detected=pooled.detected + score.detected,
                missed=pooled.missed + score.missed,
                false_alarms=pooled.false_alarms + score.false_alarms,
                split=pooled.split + score.split,
                merged=pooled.merged + score.merged,
                audio_seconds=pooled.audio_seconds + score.audio_seconds,
            )
        return VADScore(
            true_positive_frames=pooled.true_positive_frames,
            false_positive_frames=pooled.false_positive_frames,
            false_negative_frames=pooled.false_negative_frames,
            reference_segments=pooled.reference_segments,
            hypothesis_segments=pooled.hypothesis_segments,
            detected=pooled.detected,
            missed=pooled.missed,
            false_alarms=pooled.false_alarms,
            split=pooled.split,
            merged=pooled.merged,
            onset_errors_ms=tuple(onsets),
            offset_errors_ms=tuple(offsets),
            audio_seconds=pooled.audio_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": {name: score.to_dict() for name, score in self.scores.items()},
            "total": self.total.to_dict(),
        }
