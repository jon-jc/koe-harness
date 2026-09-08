"""Mock providers: determinism, controllable quality, controllable failure.

These properties are what the eval and routing layers are built on, so they are
tested as contracts rather than as incidental behaviour.
"""

from __future__ import annotations

import pytest

from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import attribute_speakers
from koe.providers.base import Message, Modality, ProviderError
from koe.providers.mock import (
    MEETING_EN,
    MEETING_JA,
    MEETING_MIXED,
    MockASR,
    MockDiarization,
    MockLLM,
)
from koe.text.script import Language


def audio(seconds: float = 24.0) -> AudioChunk:
    return AudioChunk(data=b"\x00" * STANDARD_FORMAT.bytes_for(seconds))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


async def test_degradation_is_reproducible() -> None:
    """Determinism must not depend on call order or a global RNG."""
    a = MockASR(degradation=0.2)
    b = MockASR(degradation=0.2)

    first = (await a.transcribe(audio())).text
    _ = await b.transcribe(audio())  # consume a call to perturb any shared state
    second = (await b.transcribe(audio())).text

    assert first == second


async def test_clean_provider_reproduces_the_script_exactly() -> None:
    asr = MockASR(script=MEETING_JA, degradation=0.0)
    transcript = await asr.transcribe(audio())
    assert transcript.text == "".join(u.text for u in MEETING_JA)


async def test_degradation_actually_changes_the_text() -> None:
    """A backend with a known error rate is what validates the metrics."""
    clean = await MockASR(degradation=0.0).transcribe(audio())
    noisy = await MockASR(degradation=0.3).transcribe(audio())
    assert clean.text != noisy.text


def test_reference_is_never_degraded() -> None:
    """The reference is ground truth; corrupting it would invalidate every eval."""
    asr = MockASR(script=MEETING_JA, degradation=0.5)
    assert asr.reference().text == "".join(u.text for u in MEETING_JA)


# --------------------------------------------------------------------------
# failure injection
# --------------------------------------------------------------------------


async def test_fail_first_recovers_after_the_configured_count() -> None:
    asr = MockASR(fail_first=2)

    for _ in range(2):
        with pytest.raises(ProviderError):
            await asr.transcribe(audio())

    transcript = await asr.transcribe(audio())
    assert transcript.text


async def test_always_fail_raises_a_retryable_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await MockASR(always_fail=True).transcribe(audio())
    assert excinfo.value.retryable
    assert excinfo.value.provider == "mock-asr"


# --------------------------------------------------------------------------
# provider metadata
# --------------------------------------------------------------------------


def test_info_declares_a_known_error_rate() -> None:
    info = MockASR(degradation=0.08).info
    assert info.modality is Modality.ASR
    assert info.error_rate_for(Language.JA) == pytest.approx(0.08)


def test_unknown_quality_is_treated_as_bad_not_free() -> None:
    """An unmeasured provider must not look like a safe cheap default."""
    info = MockASR().info
    assert info.error_rate_for(Language.MIXED) == pytest.approx(0.0)  # both languages declared

    from koe.providers.base import Modality as M
    from koe.providers.base import ProviderInfo

    blank = ProviderInfo(name="x", modality=M.ASR, model="y")
    assert blank.error_rate_for(Language.JA) == 1.0


def test_measurement_replaces_the_prior() -> None:
    info = MockASR().info
    assert not info.measured
    updated = info.with_measurement(Language.JA, 0.11)
    assert updated.measured
    assert updated.error_rate_for(Language.JA) == pytest.approx(0.11)


def test_mixed_language_support_requires_both_languages() -> None:
    from koe.providers.base import Modality as M
    from koe.providers.base import ProviderInfo

    ja_only = ProviderInfo(name="x", modality=M.ASR, model="y", languages=frozenset({Language.JA}))
    assert ja_only.supports(Language.JA)
    assert not ja_only.supports(Language.MIXED)
    assert MockASR().info.supports(Language.MIXED)


def test_cost_estimation() -> None:
    info = MockASR(cost_per_audio_minute_usd=0.006).info
    assert info.estimate_audio_cost(120.0) == pytest.approx(0.012)


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------


async def test_japanese_script_carries_the_hard_cases() -> None:
    """The fixture has to contain what actually breaks naive pipelines."""
    text = (await MockASR(script=MEETING_JA).transcribe(audio())).text
    assert "第三四半期" in text  # numerals in kanji
    assert "百二十" in text  # a number written additively
    assert "三月十日" in text  # a date
    assert "KPI" in text  # embedded English acronym


async def test_mixed_script_is_detected_as_code_switched() -> None:
    transcript = await MockASR(script=MEETING_MIXED).transcribe(audio())
    languages = {seg.language for seg in transcript.segments}
    assert Language.MIXED in languages or Language.JA in languages


async def test_english_script_mirrors_the_japanese_one() -> None:
    """Matched content, so bilingual behaviour is compared like for like."""
    assert len(MEETING_EN) == len(MEETING_JA)
    transcript = await MockASR(script=MEETING_EN).transcribe(audio())
    assert transcript.language is Language.EN


# --------------------------------------------------------------------------
# diarization and fusion together
# --------------------------------------------------------------------------


async def test_diarization_matches_the_script_speakers() -> None:
    diarization = await MockDiarization(script=MEETING_JA).diarize(audio())
    assert set(diarization.speakers) == {"田中", "鈴木", "佐藤"}


async def test_end_to_end_fusion_recovers_every_speaker() -> None:
    asr = MockASR(script=MEETING_JA, degradation=0.0)
    diarizer = MockDiarization(script=MEETING_JA)

    transcript = await asr.transcribe(audio())
    diarization = await diarizer.diarize(audio())
    fused = attribute_speakers(transcript, diarization)

    assert {s.speaker for s in fused.segments} == {"田中", "鈴木", "佐藤"}
    assert all(s.speaker for s in fused.segments)


async def test_boundary_error_models_real_diarization_drift() -> None:
    """Turns that do not line up with sentence ends are the common failure."""
    diarization = await MockDiarization(script=MEETING_JA, boundary_error=0.4).diarize(audio())
    exact = await MockDiarization(script=MEETING_JA).diarize(audio())
    assert [t.start for t in diarization.turns] != [t.start for t in exact.turns]


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


async def test_llm_matches_a_response_by_prompt_substring() -> None:
    llm = MockLLM(responses={"議事録": '{"summary": "ok"}'}, default_response="{}")
    reply = await llm.complete([Message(role="user", content="この議事録を作成")])
    assert reply.text == '{"summary": "ok"}'


async def test_llm_falls_back_to_the_default_response() -> None:
    llm = MockLLM(responses={"nope": "x"}, default_response="fallback")
    assert (await llm.complete([Message(role="user", content="hi")])).text == "fallback"


async def test_llm_reports_usage() -> None:
    llm = MockLLM(default_response="a response with some length")
    reply = await llm.complete([Message(role="user", content="a prompt")])
    assert reply.usage.input_tokens > 0
    assert reply.usage.output_tokens > 0
    assert reply.usage.provider == "mock-llm"


async def test_usage_adds() -> None:
    from koe.providers.base import Usage

    total = Usage(input_tokens=10, cost_usd=0.1) + Usage(input_tokens=5, cost_usd=0.2)
    assert total.input_tokens == 15
    assert total.cost_usd == pytest.approx(0.3)
