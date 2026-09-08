"""Deterministic in-process providers.

These are not throwaway test doubles. They are load-bearing infrastructure:

* **The whole system runs with no credentials.** ``pytest`` and the demo work
  on a clean checkout, which is what keeps CI free and deterministic and lets a
  reviewer see the thing work without being handed API keys.
* **They can be told to be bad, precisely.** :class:`MockASR` accepts a
  `degradation` rate and produces *deterministically* corrupted output at that
  rate. That gives the eval harness a backend with a known error rate to
  validate its own metrics against -- if a CER implementation cannot recover a
  known 8% corruption, the bug is in the metric, not the model. It also gives
  the router a realistically bad-but-cheap provider to trade against a good
  expensive one.
* **They can fail on cue.** `fail_first` and `always_fail` drive circuit
  breaker and fallback tests without waiting on a real outage.

Determinism comes from hashing the input, not from a global RNG, so results do
not depend on test execution order.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from koe.domain.audio import AudioChunk
from koe.domain.transcript import Diarization, Segment, SpeakerTurn, Transcript, Word
from koe.providers.base import (
    LLMResponse,
    Message,
    Modality,
    ProviderError,
    ProviderInfo,
    Usage,
    measured,
)
from koe.text.script import Language, primary_language


@dataclass(frozen=True, slots=True)
class ScriptedUtterance:
    """One line of a scripted meeting."""

    speaker: str
    text: str
    start: float
    end: float
    language: Language = Language.UNKNOWN

    def resolved_language(self) -> Language:
        return (
            self.language if self.language is not Language.UNKNOWN else primary_language(self.text)
        )


def _script(
    lines: Sequence[tuple[str, str]], *, seconds_each: float = 4.0
) -> list[ScriptedUtterance]:
    return [
        ScriptedUtterance(
            speaker=speaker,
            text=text,
            start=i * seconds_each,
            end=(i + 1) * seconds_each - 0.2,
        )
        for i, (speaker, text) in enumerate(lines)
    ]


#: A Japanese quarterly review. Contains the things that actually break
#: naive pipelines: numerals in three forms (第三四半期, 百二十, 三月十日),
#: an embedded English acronym, keigo, and action items with owners and dates.
MEETING_JA: list[ScriptedUtterance] = _script(
    [
        ("田中", "本日の議題は第三四半期の売上レビューです。"),
        ("鈴木", "売上は前年比で百二十パーセント、目標を達成しました。"),
        ("田中", "KPIのダッシュボードは来週までに更新できますか。"),
        ("佐藤", "はい、金曜日までに対応します。"),
        ("鈴木", "新機能のリリースは三月十日を予定しています。"),
        ("田中", "では、次回の会議は再来週の火曜日にしましょう。"),
    ]
)

#: The same meeting in English, so bilingual behaviour can be compared on
#: matched content rather than on two unrelated recordings.
MEETING_EN: list[ScriptedUtterance] = _script(
    [
        ("Alice", "Today's agenda is the Q3 revenue review."),
        ("Bob", "Revenue came in at 120% of target, up year over year."),
        ("Alice", "Can you have the KPI dashboard updated by next week?"),
        ("Carol", "Yes, I'll take care of it by Friday."),
        ("Bob", "The new feature release is planned for March 10th."),
        ("Alice", "Let's schedule the next meeting for Tuesday after next."),
    ]
)

#: Clause-level JA/EN switching, which is normal in Japanese tech meetings and
#: is where single-language ASR configurations quietly degrade.
MEETING_MIXED: list[ScriptedUtterance] = _script(
    [
        ("田中", "まずQ3のrevenueをreviewしましょう。"),
        ("Bob", "We hit 120% of target. 目標達成です。"),
        ("田中", "KPI dashboardのupdateは来週までにお願いします。"),
        ("佐藤", "はい、Fridayまでに対応します。"),
    ]
)


def _deterministic_degrade(text: str, rate: float, salt: str) -> str:
    """Corrupt `text` at approximately `rate`, reproducibly.

    Character choice is driven by a hash of (salt, position), so the same input
    always yields the same corruption regardless of test ordering or how many
    other calls happened first.
    """
    if rate <= 0 or not text:
        return text
    out: list[str] = []
    for i, ch in enumerate(text):
        digest = hashlib.sha256(f"{salt}:{i}:{ch}".encode()).digest()
        draw = digest[0] / 255.0
        if draw >= rate:
            out.append(ch)
            continue
        # Three corruption modes, matching how ASR actually fails:
        # deletion, substitution, and insertion.
        mode = digest[1] % 3
        if mode == 0:
            continue  # deletion
        if mode == 1:
            out.append(chr(0x3042 + digest[2] % 80) if ord(ch) > 0x2000 else "x")  # substitution
        else:
            out.append(ch)
            out.append(ch)  # insertion (doubling)
    return "".join(out)


def _words_for(utterance: ScriptedUtterance, text: str) -> list[Word]:
    """Spread word timings evenly across the utterance's span.

    Even spacing is a fiction, but it is a *consistent* fiction: what the fusion
    logic downstream needs is monotonic non-overlapping spans within the
    utterance, not physically accurate boundaries.
    """
    language = utterance.resolved_language()
    units = list(text) if language is Language.JA else text.split()
    if not units:
        return []
    span = max(utterance.end - utterance.start, 0.01) / len(units)
    return [
        Word(
            text=unit,
            start=utterance.start + i * span,
            end=utterance.start + (i + 1) * span,
            confidence=0.95,
        )
        for i, unit in enumerate(units)
    ]


@dataclass
class MockASR:
    """Scripted speech recognition with controllable quality and failure."""

    script: Sequence[ScriptedUtterance] = field(default_factory=lambda: MEETING_JA)
    degradation: float = 0.0
    latency_ms: float = 0.0
    fail_first: int = 0
    always_fail: bool = False
    name: str = "mock-asr"
    cost_per_audio_minute_usd: float = 0.0
    word_timestamps: bool = True
    info: ProviderInfo = field(init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.info = ProviderInfo(
            name=self.name,
            modality=Modality.ASR,
            model=f"scripted-d{self.degradation:g}",
            languages=frozenset({Language.JA, Language.EN}),
            supports_streaming=True,
            supports_word_timestamps=self.word_timestamps,
            cost_per_audio_minute_usd=self.cost_per_audio_minute_usd,
            typical_rtf=self.latency_ms / 1000.0 if self.latency_ms else 0.05,
            typical_first_result_ms=self.latency_ms,
            # A mock's error rate is known exactly, which is precisely what
            # makes it useful for validating the metrics that measure it.
            expected_error_rate={
                Language.JA: self.degradation,
                Language.EN: self.degradation,
            },
        )

    def reference(self) -> Transcript:
        """The uncorrupted transcript -- the ground truth for evaluation."""
        return self._build(degrade=False)

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise ProviderError(
                f"{self.name} is unavailable (call {self.calls})",
                provider=self.name,
                retryable=True,
            )
        usage = Usage(audio_seconds=audio.duration)
        async with measured(self.info, usage):
            if self.latency_ms:
                await asyncio.sleep(self.latency_ms / 1000.0)
        return self._build(degrade=True)

    def _build(self, *, degrade: bool) -> Transcript:
        segments: list[Segment] = []
        for utterance in self.script:
            text = (
                _deterministic_degrade(utterance.text, self.degradation, self.name)
                if degrade
                else utterance.text
            )
            segments.append(
                Segment(
                    text=text,
                    start=utterance.start,
                    end=utterance.end,
                    words=_words_for(utterance, text) if self.word_timestamps else [],
                    language=utterance.resolved_language(),
                    confidence=1.0 - self.degradation,
                    is_final=True,
                )
            )
        duration = max((u.end for u in self.script), default=0.0)
        languages = {s.language for s in segments}
        return Transcript(
            segments=segments,
            language=languages.pop() if len(languages) == 1 else Language.MIXED,
            duration=duration,
            provider=self.name,
            model=self.info.model,
        )


@dataclass
class MockDiarization:
    """Derives speaker turns from the same script the ASR mock uses."""

    script: Sequence[ScriptedUtterance] = field(default_factory=lambda: MEETING_JA)
    boundary_error: float = 0.0
    latency_ms: float = 0.0
    always_fail: bool = False
    name: str = "mock-diarization"
    info: ProviderInfo = field(init=False)

    def __post_init__(self) -> None:
        self.info = ProviderInfo(
            name=self.name,
            modality=Modality.DIARIZATION,
            model="scripted",
            languages=frozenset({Language.JA, Language.EN}),
            typical_rtf=0.05,
        )

    async def diarize(
        self,
        audio: AudioChunk,
        *,
        num_speakers: int | None = None,
    ) -> Diarization:
        if self.always_fail:
            raise ProviderError(f"{self.name} is unavailable", provider=self.name)
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)

        turns: list[SpeakerTurn] = []
        for i, utterance in enumerate(self.script):
            # A boundary shift models the most common real diarization error:
            # turns that do not line up with sentence ends.
            shift = self.boundary_error * (1 if i % 2 else -1)
            turns.append(
                SpeakerTurn(
                    speaker=utterance.speaker,
                    start=max(0.0, utterance.start + shift),
                    end=max(0.0, utterance.end + shift),
                    confidence=0.9,
                )
            )
        return Diarization(turns=turns, provider=self.name, model="scripted")


@dataclass
class MockLLM:
    """Returns canned completions, optionally keyed by a substring of the prompt."""

    responses: dict[str, str] = field(default_factory=dict)
    default_response: str = "{}"
    latency_ms: float = 0.0
    always_fail: bool = False
    fail_first: int = 0
    name: str = "mock-llm"
    info: ProviderInfo = field(init=False)
    calls: int = field(default=0, init=False)
    last_messages: list[Message] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.info = ProviderInfo(
            name=self.name,
            modality=Modality.LLM,
            model="scripted",
            languages=frozenset({Language.JA, Language.EN}),
            cost_per_1k_input_tokens_usd=0.0,
            cost_per_1k_output_tokens_usd=0.0,
            typical_first_result_ms=self.latency_ms,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls += 1
        self.last_messages = list(messages)
        if self.always_fail or self.calls <= self.fail_first:
            raise ProviderError(
                f"{self.name} is unavailable (call {self.calls})",
                provider=self.name,
                retryable=True,
            )

        usage = Usage()
        async with measured(self.info, usage):
            if self.latency_ms:
                await asyncio.sleep(self.latency_ms / 1000.0)

        joined = "\n".join(m.content for m in messages)
        text = next(
            (reply for key, reply in self.responses.items() if key in joined),
            self.default_response,
        )
        usage.input_tokens = max(1, len(joined) // 4)
        usage.output_tokens = max(1, len(text) // 4)
        return LLMResponse(text=text, usage=usage, model=self.info.model, stop_reason="end_turn")
