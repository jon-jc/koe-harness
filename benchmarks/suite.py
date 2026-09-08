"""The benchmark suite.

Covers the components whose cost actually constrains the product:

* **Text processing** runs on every utterance, twice (reference and hypothesis)
  during evaluation, and once on the realtime path.
* **Metrics** run over the whole corpus on every eval, and the bootstrap runs
  2,000 resamples on top of that — it is the slowest thing in CI by far.
* **The audio path** is where real-time factor decides whether concurrency is
  possible at all.
* **Routing** happens per request, so it has to be cheap enough to be free.
"""

from __future__ import annotations

import array
import asyncio
import math

from benchmarks.runner import Bench
from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Diarization, Segment, SpeakerTurn, Transcript, attribute_speakers
from koe.evaluation.corpus import build_corpus
from koe.evaluation.metrics import character_error_rate, diarization_error_rate, word_error_rate
from koe.evaluation.statistics import bootstrap_interval, paired_bootstrap
from koe.minutes.demo import canned_minutes
from koe.minutes.guardrails import check_minutes
from koe.pipeline.stabilizer import Stabilizer
from koe.pipeline.vad import VAD, VADConfig
from koe.providers.mock import MEETING_JA, MockASR
from koe.routing.budget import Budget, Priority
from koe.routing.router import Router
from koe.text.normalize import normalize_for_display, normalize_for_scoring
from koe.text.numbers import to_arabic
from koe.text.script import Language, detect_language
from koe.text.tokenize import tokenizer_for

bench = Bench()

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

JA_LINE = "本日は、ＫＰＩのダッシュボードを二千二十五年三月十日までに更新します。"
EN_LINE = "Today we'll review the Q3 revenue numbers and update the dashboard by Friday."
MIXED_LINE = "そのAPIのlatencyがSLOを超えているので、来週までにbenchmarkを取ります。"

JA_PARAGRAPH = JA_LINE * 20
CORPUS = build_corpus()
CORPUS_PAIRS = [(case.resolved_reference(), case.resolved_reference()) for case in CORPUS.cases]

SAMPLE_RATE = 16_000


def tone(ms: float, amplitude: int = 8000) -> bytes:
    count = int(SAMPLE_RATE * ms / 1000.0)
    return array.array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)) for i in range(count)),
    ).tobytes()


def quiet(ms: float) -> bytes:
    count = int(SAMPLE_RATE * ms / 1000.0)
    return array.array("h", ((20 if i % 7 == 0 else -20) for i in range(count))).tobytes()


ONE_SECOND_SPEECH = tone(1000)
UTTERANCE_AUDIO = quiet(300) + tone(3000) + quiet(1000)
UTTERANCE_SECONDS = STANDARD_FORMAT.duration_of(len(UTTERANCE_AUDIO))

# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


@bench.case(
    "normalize (scoring, JA)",
    group="text",
    units_per_call=len(JA_LINE),
    unit_label="char",
)
def _normalize_scoring_ja() -> None:
    normalize_for_scoring(JA_LINE)


@bench.case(
    "normalize (display, JA)",
    group="text",
    units_per_call=len(JA_LINE),
    unit_label="char",
)
def _normalize_display_ja() -> None:
    normalize_for_display(JA_LINE)


@bench.case(
    "normalize (scoring, EN)",
    group="text",
    units_per_call=len(EN_LINE),
    unit_label="char",
)
def _normalize_scoring_en() -> None:
    normalize_for_scoring(EN_LINE)


@bench.case(
    "normalize (scoring, mixed)",
    group="text",
    units_per_call=len(MIXED_LINE),
    unit_label="char",
)
def _normalize_scoring_mixed() -> None:
    normalize_for_scoring(MIXED_LINE)


@bench.case(
    "normalize (scoring, 1.4k chars)",
    group="text",
    units_per_call=len(JA_PARAGRAPH),
    unit_label="char",
    repeats=50,
)
def _normalize_paragraph() -> None:
    normalize_for_scoring(JA_PARAGRAPH)


@bench.case(
    "detect_language (mixed)", group="text", units_per_call=len(MIXED_LINE), unit_label="char"
)
def _detect() -> None:
    detect_language(MIXED_LINE)


@bench.case("漢数字 -> arabic", group="text", units_per_call=len(JA_LINE), unit_label="char")
def _numerals() -> None:
    to_arabic(JA_LINE)


@bench.case("tokenize JA", group="text", units_per_call=len(JA_LINE), unit_label="char")
def _tokenize_ja() -> None:
    tokenizer_for(Language.JA).tokenize(JA_LINE)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


@bench.case("CER (one utterance)", group="metrics", units_per_call=len(JA_LINE), unit_label="char")
def _cer() -> None:
    character_error_rate(JA_LINE, JA_LINE.replace("三月", "四月"))


@bench.case("WER (one utterance)", group="metrics", units_per_call=len(EN_LINE), unit_label="char")
def _wer() -> None:
    word_error_rate(EN_LINE, EN_LINE.replace("Friday", "Monday"))


@bench.case(
    "CER (41-case corpus)",
    group="metrics",
    units_per_call=len(CORPUS_PAIRS),
    unit_label="case",
    repeats=30,
)
def _cer_corpus() -> None:
    for reference, hypothesis in CORPUS_PAIRS:
        character_error_rate(reference, hypothesis)


REF_DIAR = Diarization(
    turns=[SpeakerTurn(speaker=f"S{i % 3}", start=i * 4.0, end=i * 4.0 + 3.5) for i in range(15)]
)
HYP_DIAR = Diarization(
    turns=[
        SpeakerTurn(speaker=f"X{i % 3}", start=i * 4.0 + 0.2, end=i * 4.0 + 3.6) for i in range(15)
    ]
)


@bench.case("DER (60s, 3 speakers)", group="metrics", audio_seconds_per_call=60.0, repeats=30)
def _der() -> None:
    diarization_error_rate(REF_DIAR, HYP_DIAR)


SAMPLES = [character_error_rate(r, h) for r, h in CORPUS_PAIRS]


@bench.case("bootstrap CI (2000 resamples)", group="metrics", repeats=5)
def _bootstrap() -> None:
    bootstrap_interval(SAMPLES, iterations=2000)


@bench.case("paired bootstrap + permutation", group="metrics", repeats=3)
def _paired() -> None:
    paired_bootstrap(SAMPLES, SAMPLES, iterations=1000)


# --------------------------------------------------------------------------
# audio path
# --------------------------------------------------------------------------


@bench.case(
    "VAD (1s of audio)",
    group="audio path",
    audio_seconds_per_call=1.0,
    units_per_call=1.0,
    unit_label="s audio",
)
def _vad() -> None:
    VAD(config=VADConfig()).push(ONE_SECOND_SPEECH)


@bench.case("stabilizer update", group="audio path", repeats=500)
def _stabilize() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.JA)
    stabilizer.update("本日の議題は")
    stabilizer.update("本日の議題は売上")


TRANSCRIPT = Transcript(
    segments=[
        Segment(
            text=u.text,
            start=u.start,
            end=u.end,
            language=Language.JA,
            words=[],
        )
        for u in MEETING_JA
    ],
    language=Language.JA,
)
DIAR = Diarization(
    turns=[SpeakerTurn(speaker=u.speaker, start=u.start, end=u.end) for u in MEETING_JA]
)


@bench.case(
    "ASR x diarization fusion",
    group="audio path",
    audio_seconds_per_call=max(u.end for u in MEETING_JA),
    repeats=100,
)
def _fusion() -> None:
    attribute_speakers(TRANSCRIPT, DIAR)


AUDIO_CHUNK = AudioChunk(data=UTTERANCE_AUDIO)
_ASR = MockASR(script=MEETING_JA[:1], degradation=0.0)


@bench.case(
    "mock ASR transcribe (utterance)",
    group="audio path",
    audio_seconds_per_call=UTTERANCE_SECONDS,
    repeats=100,
)
def _asr() -> None:
    asyncio.run(_ASR.transcribe(AUDIO_CHUNK))


# --------------------------------------------------------------------------
# routing and guardrails
# --------------------------------------------------------------------------

_ROUTER: Router[MockASR] = Router(
    [
        MockASR(name=f"p{i}", degradation=0.02 * i, cost_per_audio_minute_usd=0.002 * i)
        for i in range(1, 6)
    ]
)
_BUDGET = Budget(priority=Priority.BALANCED, language=Language.JA)


@bench.case("router.select (5 providers)", group="routing & guardrails", repeats=500)
def _select() -> None:
    _ROUTER.select(_BUDGET)


_MINUTES = canned_minutes()
_TRANSCRIPT_TEXT = "\n".join(f"{u.speaker}: {u.text}" for u in MEETING_JA)


@bench.case("groundedness check (4 claims)", group="routing & guardrails", repeats=200)
def _groundedness() -> None:
    check_minutes(_MINUTES, _TRANSCRIPT_TEXT)
