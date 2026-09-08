"""The evaluation runner and its report.

Runs a dataset through an ASR backend (and optionally a diarizer), scores every
case, and produces a report sliced by language and by tag.

The slicing is the design decision worth defending. A single corpus-level CER
tells you almost nothing you can act on. Slicing by tag turns the same run into
findings: *code-switched utterances are 3x worse than monolingual ones*, or
*numeral-heavy cases regressed while everything else held*. Those are
statements an engineer can do something about, and they are invisible in an
aggregate.

Cost and latency are collected alongside quality on the same run, because they
are not separable concerns -- "which model should we use" is always a question
about all three at once, and measuring them in different places guarantees they
get traded off by anecdote.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Diarization, Transcript
from koe.evaluation.dataset import Dataset, EvalCase
from koe.evaluation.metrics import (
    DiarizationScore,
    ErrorRate,
    TranscriptionScore,
    diarization_error_rate,
    score_transcription,
    speaker_count_error,
)
from koe.evaluation.statistics import MetricSummary, summarize
from koe.providers.base import ProviderError, Usage
from koe.text.normalize import Normalizer
from koe.text.script import Language

logger = logging.getLogger(__name__)


class TranscribeFn(Protocol):
    """Anything that turns a case into a transcript.

    Deliberately narrower than ``ASRProvider``: an evaluation target is often a
    whole pipeline (router + fallback + post-processing), not a bare provider,
    and the harness must be able to score that as one unit.

    May return either a :class:`Transcript` or a ``(Transcript, Usage)`` pair.
    The pair form is how cost reaches the report -- quality and cost have to be
    measured on the *same run*, or they end up traded off against each other by
    anecdote rather than by data.
    """

    async def __call__(self, case: EvalCase) -> Transcript | tuple[Transcript, Usage]: ...


@dataclass(slots=True)
class CaseResult:
    """Everything measured for one case."""

    case_id: str
    language: Language
    tags: list[str]
    score: TranscriptionScore | None = None
    diarization: DiarizationScore | None = None
    speaker_delta: int = 0
    usage: Usage = field(default_factory=Usage)
    wall_ms: float = 0.0
    audio_seconds: float = 0.0
    hypothesis: str = ""
    reference: str = ""
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def rtf(self) -> float:
        """Real-time factor: processing seconds per audio second."""
        if self.audio_seconds <= 0:
            return 0.0
        return (self.wall_ms / 1000.0) / self.audio_seconds


@dataclass(slots=True)
class Slice:
    """Scored results for one subset of the data."""

    name: str
    cer: MetricSummary
    wer: MetricSummary
    n: int
    der: float | None = None

    def __str__(self) -> str:
        der = f" DER {self.der:.2%}" if self.der is not None else ""
        return f"{self.name:<24} n={self.n:<4} {self.cer.interval}{der}"


@dataclass(slots=True)
class EvalReport:
    """The result of one evaluation run."""

    system: str
    dataset: str
    results: list[CaseResult] = field(default_factory=list)
    seed: int = 0

    # -- aggregate views -----------------------------------------------------

    @property
    def succeeded(self) -> list[CaseResult]:
        return [r for r in self.results if not r.failed and r.score is not None]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if r.failed]

    @property
    def failure_rate(self) -> float:
        """Share of cases that errored.

        Reported separately from quality on purpose: a backend that crashes on
        20% of inputs and is excellent on the rest has a *reliability* problem,
        and averaging that into a quality score would disguise it.
        """
        if not self.results:
            return 0.0
        return len(self.failures) / len(self.results)

    def cer_samples(self) -> list[ErrorRate]:
        return [r.score.cer for r in self.succeeded if r.score]

    def wer_samples(self) -> list[ErrorRate]:
        return [r.score.wer for r in self.succeeded if r.score]

    @property
    def total_cost_usd(self) -> float:
        return sum(r.usage.cost_usd for r in self.results)

    @property
    def total_audio_seconds(self) -> float:
        return sum(r.audio_seconds for r in self.results)

    @property
    def cost_per_audio_hour(self) -> float:
        """The unit that actually appears on an invoice."""
        hours = self.total_audio_seconds / 3600.0
        return self.total_cost_usd / hours if hours > 0 else 0.0

    def latency_percentile(self, fraction: float) -> float:
        values = sorted(r.wall_ms for r in self.succeeded)
        if not values:
            return 0.0
        index = min(len(values) - 1, int(fraction * len(values)))
        return values[index]

    @property
    def mean_rtf(self) -> float:
        values = [r.rtf for r in self.succeeded if r.rtf > 0]
        return sum(values) / len(values) if values else 0.0

    # -- slicing -------------------------------------------------------------

    def slice_by_language(self, *, seed: int = 0) -> list[Slice]:
        out: list[Slice] = []
        for language in sorted({r.language for r in self.succeeded}):
            subset = [r for r in self.succeeded if r.language is language]
            out.append(self._make_slice(f"lang:{language.value}", subset, seed=seed))
        return out

    def slice_by_tag(self, *, seed: int = 0) -> list[Slice]:
        tags = sorted({tag for r in self.succeeded for tag in r.tags})
        out: list[Slice] = []
        for tag in tags:
            subset = [r for r in self.succeeded if tag in r.tags]
            out.append(self._make_slice(f"tag:{tag}", subset, seed=seed))
        return out

    def overall(self, *, seed: int = 0) -> Slice:
        return self._make_slice("overall", self.succeeded, seed=seed)

    def _make_slice(self, name: str, subset: Sequence[CaseResult], *, seed: int) -> Slice:
        cers = [r.score.cer for r in subset if r.score]
        wers = [r.score.wer for r in subset if r.score]
        ders = [r.diarization.value for r in subset if r.diarization is not None]
        return Slice(
            name=name,
            cer=summarize("CER", cers, seed=seed),
            wer=summarize("WER", wers, seed=seed),
            n=len(subset),
            der=sum(ders) / len(ders) if ders else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary, for persisting a baseline."""
        overall = self.overall(seed=self.seed)
        return {
            "system": self.system,
            "dataset": self.dataset,
            "n_cases": len(self.results),
            "n_failed": len(self.failures),
            "failure_rate": self.failure_rate,
            "cer": overall.cer.interval.point,
            "cer_lower": overall.cer.interval.lower,
            "cer_upper": overall.cer.interval.upper,
            "wer": overall.wer.interval.point,
            "wer_tokenizer": overall.wer.tokenizer,
            "der": overall.der,
            "cost_usd": self.total_cost_usd,
            "cost_per_audio_hour": self.cost_per_audio_hour,
            "latency_p50_ms": self.latency_percentile(0.50),
            "latency_p95_ms": self.latency_percentile(0.95),
            "mean_rtf": self.mean_rtf,
            "slices": {
                s.name: {"n": s.n, "cer": s.cer.interval.point, "wer": s.wer.interval.point}
                for s in (
                    *self.slice_by_language(seed=self.seed),
                    *self.slice_by_tag(seed=self.seed),
                )
            },
        }


async def _score_case(
    case: EvalCase,
    transcribe: TranscribeFn,
    *,
    diarize: Any | None,
    normalizer: Normalizer | None,
    der_collar: float,
) -> CaseResult:
    result = CaseResult(
        case_id=case.id,
        language=case.resolved_language(),
        tags=list(case.tags),
        reference=case.resolved_reference(),
        audio_seconds=case.duration(),
    )

    started = time.perf_counter()
    try:
        outcome = await transcribe(case)
    except ProviderError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.wall_ms = (time.perf_counter() - started) * 1000.0
        logger.warning("case %s failed: %s", case.id, exc)
        return result
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.wall_ms = (time.perf_counter() - started) * 1000.0
        logger.exception("case %s raised", case.id)
        return result

    result.wall_ms = (time.perf_counter() - started) * 1000.0

    if isinstance(outcome, tuple):
        transcript, result.usage = outcome
    else:
        transcript = outcome

    result.hypothesis = transcript.text
    result.score = score_transcription(
        result.reference,
        result.hypothesis,
        language=result.language,
        normalizer=normalizer,
    )

    if diarize is not None and case.speakers:
        try:
            hypothesis: Diarization = await diarize(case)
        except Exception as exc:  # noqa: BLE001
            logger.warning("diarization failed for %s: %s", case.id, exc)
        else:
            reference = Diarization(turns=case.speakers)
            result.diarization = diarization_error_rate(reference, hypothesis, collar=der_collar)
            result.speaker_delta = speaker_count_error(reference, hypothesis)

    return result


async def run_evaluation(
    dataset: Dataset,
    transcribe: TranscribeFn,
    *,
    system: str = "system",
    diarize: Any | None = None,
    normalizer: Normalizer | None = None,
    concurrency: int = 4,
    der_collar: float = 0.0,
    seed: int = 0,
) -> EvalReport:
    """Score `dataset` against `transcribe`.

    Cases run concurrently up to `concurrency`. A failing case is recorded as a
    failure and the run continues -- an eval that aborts on the first error
    tells you nothing about the other thirty-nine cases, which is exactly when
    you most need the information.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(case: EvalCase) -> CaseResult:
        async with semaphore:
            return await _score_case(
                case,
                transcribe,
                diarize=diarize,
                normalizer=normalizer,
                der_collar=der_collar,
            )

    results = await asyncio.gather(*(guarded(case) for case in dataset.cases))
    return EvalReport(system=system, dataset=dataset.name, results=list(results), seed=seed)


def mock_transcriber(asr: Any) -> TranscribeFn:
    """Adapt a scripted ASR mock into a :class:`TranscribeFn`.

    Rebuilds the provider per case so each one is scored against its own script
    rather than whatever the mock was constructed with.
    """
    from koe.providers.mock import MockASR

    async def transcribe(case: EvalCase) -> Transcript | tuple[Transcript, Usage]:
        utterances = case.utterances()
        provider = MockASR(
            script=utterances or asr.script,
            degradation=asr.degradation,
            latency_ms=asr.latency_ms,
            name=asr.name,
            cost_per_audio_minute_usd=asr.cost_per_audio_minute_usd,
        )
        seconds = case.duration() or 1.0
        audio = AudioChunk(data=b"\x00" * STANDARD_FORMAT.bytes_for(seconds))
        transcript = await provider.transcribe(audio)
        usage = Usage(
            provider=provider.info.name,
            model=provider.info.model,
            audio_seconds=seconds,
            cost_usd=provider.info.estimate_audio_cost(seconds),
        )
        return (transcript, usage)

    return transcribe
