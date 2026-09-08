"""ASR and diarization error metrics.

Three decisions in here shape every number the harness reports.

**1. Japanese is scored by CER, not WER.** Japanese has no word spaces, so
"words" only exist relative to a segmenter. Two teams reporting WER on the same
Japanese audio with different tokenizers have not measured the same thing. CER
has no such dependency, which is why published Japanese ASR results use it.
WER is still computed and reported for Japanese, but always **carrying the name
of the tokenizer that produced it** -- a Japanese WER without that label is not
a comparable number.

**2. Error rates are not clamped to 1.0.** A model that hallucinates a
paragraph onto a three-word utterance has an error rate above 100%, and that is
the correct, informative answer. Clamping would hide the single most important
failure mode of a modern ASR model: fluent, confident invention on silence.

**3. Alignment is computed with linear memory.** A one-hour meeting is ~50k
Japanese characters; a full Levenshtein matrix against another 50k-character
hypothesis is 2.5 billion cells. The two-row formulation carries the
substitution/deletion/insertion counts forward with the cost, so the error
breakdown survives without the matrix. The full backtrace is available
separately, for short spans where seeing the actual alignment is worth it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from koe.domain.transcript import Diarization, SpeakerTurn
from koe.text.normalize import Normalizer, normalize_for_scoring
from koe.text.script import Language, primary_language
from koe.text.tokenize import surfaces, tokenizer_for


class Unit(StrEnum):
    """What an error rate counts."""

    CHARACTER = "char"
    WORD = "word"


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """An error rate with the breakdown and provenance behind it.

    `tokenizer` is part of the metric's identity, not decoration: the same audio
    scored with different segmenters yields different WERs, so a number without
    its tokenizer cannot be compared to another.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    unit: Unit
    tokenizer: str = ""

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def value(self) -> float:
        """Errors divided by reference length.

        Deliberately unclamped -- see the module docstring. An empty reference
        with a non-empty hypothesis is defined as 1.0 per insertion, since
        dividing by zero has no useful answer but "the model invented text on
        silence" does.
        """
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else float(self.insertions)
        return self.errors / self.reference_length

    @property
    def accuracy(self) -> float:
        return 1.0 - self.value

    @property
    def hits(self) -> int:
        return max(0, self.reference_length - self.substitutions - self.deletions)

    def __add__(self, other: ErrorRate) -> ErrorRate:
        """Pool two results.

        Corpus-level error rate is the pooled total, *not* the mean of
        per-utterance rates. Averaging rates weights a two-word utterance the
        same as a two-hundred-word one and produces a number that moves when
        the segmentation changes but the audio does not.
        """
        if self.unit is not other.unit:
            raise ValueError(f"cannot pool {self.unit} with {other.unit}")
        return ErrorRate(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_length=self.reference_length + other.reference_length,
            unit=self.unit,
            tokenizer=self.tokenizer or other.tokenizer,
        )

    def __str__(self) -> str:
        label = "CER" if self.unit is Unit.CHARACTER else "WER"
        suffix = f" [{self.tokenizer}]" if self.tokenizer else ""
        return (
            f"{label} {self.value:.2%}{suffix} "
            f"(S={self.substitutions} D={self.deletions} I={self.insertions} "
            f"N={self.reference_length})"
        )


ZERO_CER = ErrorRate(0, 0, 0, 0, Unit.CHARACTER)


def _counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Levenshtein S/D/I counts in O(min(len)) memory.

    Each cell carries ``(cost, substitutions, deletions, insertions)`` so the
    breakdown travels with the cost and no matrix has to be retained.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)

    # row j = cost of turning ref[:0] into hyp[:j] -> j insertions
    previous: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]

    for i in range(1, n + 1):
        # first column: i deletions
        current: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        ref_token = ref[i - 1]
        for j in range(1, m + 1):
            if ref_token == hyp[j - 1]:
                current.append(previous[j - 1])
                continue
            sub_cost, sub_s, sub_d, sub_i = previous[j - 1]
            del_cost, del_s, del_d, del_i = previous[j]
            ins_cost, ins_s, ins_d, ins_i = current[j - 1]
            # Ties resolve substitution > deletion > insertion. The choice is
            # arbitrary for the total but must be *fixed*, or the S/D/I split
            # would drift between runs on identical input.
            best = min(sub_cost, del_cost, ins_cost) + 1
            if sub_cost + 1 == best:
                current.append((best, sub_s + 1, sub_d, sub_i))
            elif del_cost + 1 == best:
                current.append((best, del_s, del_d + 1, del_i))
            else:
                current.append((best, ins_s, ins_d, ins_i + 1))
        previous = current

    _, substitutions, deletions, insertions = previous[m]
    return (substitutions, deletions, insertions)


@dataclass(frozen=True, slots=True)
class AlignmentOp:
    """One step of an alignment, for error analysis."""

    op: str  # equal | sub | del | ins
    reference: str | None
    hypothesis: str | None


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[AlignmentOp]:
    """Full alignment trace.

    Uses the quadratic-memory formulation, so it is intended for inspecting a
    single utterance -- showing a reviewer *which* words a model got wrong --
    rather than for scoring a corpus.
    """
    n, m = len(ref), len(hyp)
    matrix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        matrix[i][0] = i
    for j in range(m + 1):
        matrix[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = 1 + min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1])

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(AlignmentOp("equal", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + 1:
            ops.append(AlignmentOp("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and matrix[i][j] == matrix[i - 1][j] + 1:
            ops.append(AlignmentOp("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(AlignmentOp("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# text metrics
# --------------------------------------------------------------------------


def character_error_rate(
    reference: str,
    hypothesis: str,
    *,
    normalizer: Normalizer | None = None,
) -> ErrorRate:
    """CER, the primary metric for Japanese.

    Both sides are normalized with the scoring profile first, so formatting
    differences -- width, punctuation, numeral spelling -- do not show up as
    recognition errors.
    """
    normalize = normalizer.normalize if normalizer else normalize_for_scoring
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    substitutions, deletions, insertions = _counts(list(ref), list(hyp))
    return ErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(ref),
        unit=Unit.CHARACTER,
        tokenizer="character",
    )


def word_error_rate(
    reference: str,
    hypothesis: str,
    *,
    language: Language | None = None,
    normalizer: Normalizer | None = None,
) -> ErrorRate:
    """WER, the primary metric for English and a secondary one for Japanese.

    The tokenizer's name is recorded on the result. For Japanese that label is
    load-bearing: the same audio scored with a different segmenter produces a
    different WER.
    """
    normalize = normalizer.normalize if normalizer else normalize_for_scoring
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    lang = language or primary_language(ref or hypothesis)
    tokenizer = tokenizer_for(lang)
    ref_tokens = surfaces(tokenizer.tokenize(ref))
    hyp_tokens = surfaces(tokenizer.tokenize(hyp))
    substitutions, deletions, insertions = _counts(ref_tokens, hyp_tokens)
    return ErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(ref_tokens),
        unit=Unit.WORD,
        tokenizer=tokenizer.name,
    )


@dataclass(frozen=True, slots=True)
class TranscriptionScore:
    """Both metrics plus which one is authoritative for this language."""

    language: Language
    cer: ErrorRate
    wer: ErrorRate

    @property
    def primary(self) -> ErrorRate:
        """CER for Japanese, WER for English."""
        return self.cer if self.language in (Language.JA, Language.MIXED) else self.wer

    @property
    def value(self) -> float:
        return self.primary.value

    def __str__(self) -> str:
        marker = "*" if self.language in (Language.JA, Language.MIXED) else " "
        return f"[{self.language.value}] {marker}{self.cer} | {self.wer}"


def score_transcription(
    reference: str,
    hypothesis: str,
    *,
    language: Language | None = None,
    normalizer: Normalizer | None = None,
) -> TranscriptionScore:
    """Score a hypothesis against a reference, choosing the right primary metric."""
    lang = language or primary_language(reference)
    return TranscriptionScore(
        language=lang,
        cer=character_error_rate(reference, hypothesis, normalizer=normalizer),
        wer=word_error_rate(reference, hypothesis, language=lang, normalizer=normalizer),
    )


# --------------------------------------------------------------------------
# diarization metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiarizationScore:
    """Diarization Error Rate and its three components.

    DER = (missed speech + false alarm + speaker confusion) / total reference
    speech. Reporting the components separately matters because they have
    different product consequences: missed speech loses content, while speaker
    confusion attributes a real sentence to the wrong person -- which is the
    one that damages a meeting record.
    """

    missed: float
    false_alarm: float
    confusion: float
    total_speech: float
    mapping: dict[str, str]

    @property
    def value(self) -> float:
        if self.total_speech <= 0:
            return 0.0
        return (self.missed + self.false_alarm + self.confusion) / self.total_speech

    def __str__(self) -> str:
        return (
            f"DER {self.value:.2%} (missed={self.missed:.1f}s "
            f"fa={self.false_alarm:.1f}s conf={self.confusion:.1f}s "
            f"of {self.total_speech:.1f}s)"
        )


def _timeline(turns: Sequence[SpeakerTurn], resolution: float) -> dict[int, str]:
    """Quantize turns into frames of `resolution` seconds.

    Frame-level scoring is how DER is defined in the literature and it sidesteps
    the interval-arithmetic edge cases that make span-based implementations
    subtly wrong at overlaps.
    """
    frames: dict[int, str] = {}
    for turn in turns:
        start = round(turn.start / resolution)
        end = round(turn.end / resolution)
        for frame in range(start, end):
            frames[frame] = turn.speaker
    return frames


def _best_mapping(
    reference: dict[int, str],
    hypothesis: dict[int, str],
    ref_speakers: list[str],
    hyp_speakers: list[str],
) -> dict[str, str]:
    """Map hypothesis labels onto reference labels to maximise agreement.

    Diarization labels are arbitrary -- ``speaker_0`` carries no meaning -- so
    DER is only defined after an optimal one-to-one mapping. Overlap counts are
    computed once, then assigned greedily by descending overlap, which is exact
    whenever one hypothesis label dominates each reference speaker (the normal
    case) and near-optimal otherwise. An exact solver would need the Hungarian
    algorithm; the cost is not justified for meetings, where speaker counts are
    small and the greedy choice is almost always the same one.
    """
    overlap: dict[tuple[str, str], int] = {}
    for frame, ref_speaker in reference.items():
        hyp_speaker = hypothesis.get(frame)
        if hyp_speaker is not None:
            key = (ref_speaker, hyp_speaker)
            overlap[key] = overlap.get(key, 0) + 1

    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    used_hyp: set[str] = set()
    for (ref_speaker, hyp_speaker), _ in sorted(
        overlap.items(), key=lambda kv: kv[1], reverse=True
    ):
        if ref_speaker in used_ref or hyp_speaker in used_hyp:
            continue
        mapping[hyp_speaker] = ref_speaker
        used_ref.add(ref_speaker)
        used_hyp.add(hyp_speaker)

    # Unmatched hypothesis speakers map to nothing and count as confusion.
    for hyp_speaker in hyp_speakers:
        mapping.setdefault(hyp_speaker, "")
    return mapping


def diarization_error_rate(
    reference: Diarization,
    hypothesis: Diarization,
    *,
    resolution: float = 0.01,
    collar: float = 0.0,
) -> DiarizationScore:
    """Compute DER between a reference and hypothesis diarization.

    `collar` forgives errors within that many seconds of a reference boundary.
    Human annotators disagree about exact turn boundaries by more than a
    typical system does, so scoring boundaries to the millisecond measures
    annotation noise; 0.25s is the conventional value where a collar is used.
    Default here is 0.0 -- strict -- so that enabling forgiveness is a visible
    choice rather than a hidden default that flatters the numbers.
    """
    ref_frames = _timeline(reference.turns, resolution)
    hyp_frames = _timeline(hypothesis.turns, resolution)

    if collar > 0:
        forgiven: set[int] = set()
        width = round(collar / resolution)
        for turn in reference.turns:
            for boundary in (turn.start, turn.end):
                center = round(boundary / resolution)
                forgiven.update(range(center - width, center + width + 1))
        ref_frames = {f: s for f, s in ref_frames.items() if f not in forgiven}
        hyp_frames = {f: s for f, s in hyp_frames.items() if f not in forgiven}

    mapping = _best_mapping(ref_frames, hyp_frames, reference.speakers, hypothesis.speakers)

    missed = 0
    false_alarm = 0
    confusion = 0

    for frame, ref_speaker in ref_frames.items():
        hyp_speaker = hyp_frames.get(frame)
        if hyp_speaker is None:
            missed += 1
        elif mapping.get(hyp_speaker) != ref_speaker:
            confusion += 1

    for frame in hyp_frames:
        if frame not in ref_frames:
            false_alarm += 1

    return DiarizationScore(
        missed=missed * resolution,
        false_alarm=false_alarm * resolution,
        confusion=confusion * resolution,
        total_speech=len(ref_frames) * resolution,
        mapping=mapping,
    )


def speaker_count_error(reference: Diarization, hypothesis: Diarization) -> int:
    """Signed difference in speaker count.

    Tracked separately from DER because it is the error users notice first: a
    four-person meeting rendered as two speakers is obviously broken in a way
    that a 12% DER does not communicate.
    """
    return hypothesis.num_speakers - reference.num_speakers
