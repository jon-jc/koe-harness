"""Realtime pipeline: endpointing, stabilization, and session orchestration."""

from __future__ import annotations

import array
import math

import pytest

from koe.domain.audio import STANDARD_FORMAT
from koe.kernel.context import Context
from koe.pipeline.session import PartialEvent, SessionConfig, StreamingSession
from koe.pipeline.stabilizer import Stabilizer, common_prefix
from koe.pipeline.vad import VAD, SpeechState, VADConfig, frame_energy_db
from koe.providers.mock import MEETING_JA, MockASR
from koe.text.script import Language

SAMPLE_RATE = 16_000


def tone(ms: float, amplitude: int = 8000, freq: float = 220.0) -> bytes:
    """Synthetic voiced audio."""
    count = int(SAMPLE_RATE * ms / 1000.0)
    samples = array.array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(count)),
    )
    return samples.tobytes()


def quiet(ms: float, amplitude: int = 20) -> bytes:
    """Room tone -- not digital silence, which no microphone produces."""
    count = int(SAMPLE_RATE * ms / 1000.0)
    samples = array.array("h", ((amplitude if i % 7 == 0 else -amplitude) for i in range(count)))
    return samples.tobytes()


# --------------------------------------------------------------------------
# energy
# --------------------------------------------------------------------------


def test_energy_of_silence_is_finite() -> None:
    """An infinity here propagates into every threshold comparison."""
    assert frame_energy_db(array.array("h", [0] * 320)) == -100.0
    assert frame_energy_db(array.array("h")) == -100.0


def test_louder_audio_measures_higher() -> None:
    soft = frame_energy_db(array.array("h", [100] * 320))
    loud = frame_energy_db(array.array("h", [10_000] * 320))
    assert loud > soft


# --------------------------------------------------------------------------
# VAD / endpointing
# --------------------------------------------------------------------------


def test_speech_is_detected_and_endpointed() -> None:
    vad = VAD(config=VADConfig(silence_to_end_ms=300.0))
    vad.push(quiet(300))

    vad.push(tone(600))
    assert vad.state is SpeechState.SPEECH

    segments = vad.push(quiet(500))

    assert len(segments) == 1
    assert segments[0].reason == "endpoint"
    assert segments[0].duration == pytest.approx(0.6, abs=0.15)


def test_a_brief_blip_does_not_open_an_utterance() -> None:
    """A cough or a door should not become a transcription request."""
    vad = VAD(config=VADConfig(silence_to_end_ms=200.0, min_utterance_ms=300.0))
    vad.push(quiet(300))
    vad.push(tone(80))
    segments = vad.push(quiet(400))
    assert segments == []


def test_a_pause_inside_speech_does_not_end_the_utterance() -> None:
    """Speakers pause mid-sentence; cutting there truncates the verb."""
    vad = VAD(config=VADConfig(silence_to_end_ms=700.0))
    vad.push(quiet(300))
    vad.push(tone(400))
    segments = vad.push(quiet(300))  # shorter than the endpoint window
    assert segments == []
    assert vad.state is SpeechState.SPEECH

    segments = vad.push(tone(400) + quiet(900))
    assert len(segments) == 1


def test_the_noise_floor_adapts_to_a_loud_room() -> None:
    """A fixed threshold works in a quiet room and fails in every real one."""
    vad = VAD()
    vad.push(quiet(500, amplitude=2000))  # noisy room
    assert vad.noise_floor_db > -60.0
    # loud room tone alone must not read as continuous speech
    assert vad.state is SpeechState.SILENCE


def test_the_noise_floor_does_not_adapt_during_speech() -> None:
    """Otherwise the floor climbs to meet the speaker and the detector goes deaf."""
    vad = VAD(config=VADConfig(silence_to_end_ms=5_000.0))
    vad.push(quiet(300))
    floor_before = vad.noise_floor_db
    vad.push(tone(2_000))
    assert vad.noise_floor_db == pytest.approx(floor_before, abs=1.0)
    assert vad.state is SpeechState.SPEECH


def test_a_very_long_utterance_is_cut_so_results_still_arrive() -> None:
    vad = VAD(config=VADConfig(max_utterance_ms=500.0, silence_to_end_ms=5_000.0))
    vad.push(quiet(200))
    segments = vad.push(tone(1_200))
    assert segments
    assert segments[0].reason == "max-duration"


def test_flush_closes_an_open_utterance() -> None:
    """Without this, whatever was said just before hanging up is lost."""
    vad = VAD()
    vad.push(quiet(300))
    vad.push(tone(600))

    segment = vad.flush()

    assert segment is not None
    assert segment.reason == "flush"


def test_flush_on_silence_returns_nothing() -> None:
    vad = VAD()
    vad.push(quiet(500))
    assert vad.flush() is None


def test_japanese_gets_a_longer_endpoint_window() -> None:
    """JA speakers pause before sentence-final particles; EN tuning cuts there."""
    ja = VADConfig.for_language(Language.JA)
    en = VADConfig.for_language(Language.EN)
    assert ja.silence_to_end_ms > en.silence_to_end_ms


# --------------------------------------------------------------------------
# stabilization
# --------------------------------------------------------------------------


def test_common_prefix() -> None:
    assert common_prefix([["a", "b", "c"], ["a", "b", "d"]]) == ["a", "b"]
    assert common_prefix([["a"], ["b"]]) == []
    assert common_prefix([]) == []


def test_nothing_commits_before_agreement() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.JA)
    result = stabilizer.update("こんにちは")
    assert result.committed == ""
    assert result.pending == "こんにちは"


def test_an_agreed_prefix_commits() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.JA)
    stabilizer.update("本日の議題は")
    result = stabilizer.update("本日の議題は売上")

    assert result.committed.startswith("本日")
    assert "売上" in result.pending or "売上" in result.committed


def test_committed_text_is_never_revised() -> None:
    """Taking back text already shown as stable is worse than having waited."""
    stabilizer = Stabilizer(agreement=2, language=Language.EN)
    stabilizer.update("the quick brown")
    stabilizer.update("the quick brown fox")
    committed = stabilizer.committed_text
    assert committed

    # the model changes its mind about the beginning
    stabilizer.update("a slow green turtle")
    stabilizer.update("a slow green turtle indeed")

    assert stabilizer.committed_text.startswith(committed)


def test_a_disagreeing_hypothesis_commits_nothing_new() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.EN)
    stabilizer.update("hello world")
    stabilizer.update("goodbye world")
    assert stabilizer.committed_text == ""


def test_finalize_commits_everything() -> None:
    """At an endpoint the ASR has seen the whole utterance; waiting adds latency."""
    stabilizer = Stabilizer(agreement=2, language=Language.JA)
    stabilizer.update("本日の")
    result = stabilizer.finalize("本日の議題は売上レビューです")

    assert result.committed == "本日の議題は売上レビューです"
    assert result.pending == ""


def test_japanese_rejoins_without_spaces() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.JA)
    result = stabilizer.finalize("本日の議題")
    assert " " not in result.committed


def test_english_rejoins_with_spaces() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.EN)
    result = stabilizer.finalize("today's agenda is the review")
    assert result.committed == "today's agenda is the review"


def test_reset_clears_state_between_utterances() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.EN)
    stabilizer.update("one two")
    stabilizer.update("one two three")
    assert stabilizer.committed_text

    stabilizer.reset()

    assert stabilizer.committed_text == ""


def test_full_text_combines_committed_and_pending() -> None:
    stabilizer = Stabilizer(agreement=2, language=Language.EN)
    stabilizer.update("hello there")
    result = stabilizer.update("hello there world")
    assert result.full == "hello there world"


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


async def test_a_session_produces_final_segments_and_events() -> None:
    ctx = Context()
    events: list[str] = []
    for name in ("session.start", "speech.start", "asr.final", "speech.end", "session.end"):
        ctx.on(name, lambda _payload, n=name: events.append(n))

    asr = MockASR(script=MEETING_JA[:1], degradation=0.0)
    session = StreamingSession(
        ctx,
        asr,
        config=SessionConfig(
            language=Language.JA,
            emit_partials=False,
            vad=VADConfig(silence_to_end_ms=300.0),
        ),
    )

    async with session:
        await session.push_audio(quiet(300))
        await session.push_audio(tone(800))
        await session.push_audio(quiet(500))

    assert "session.start" in events
    assert "speech.start" in events
    assert "asr.final" in events
    assert "session.end" in events
    assert session.transcript().final_segments


async def test_partials_are_emitted_while_speaking() -> None:
    ctx = Context()
    partials: list[PartialEvent] = []
    ctx.on("asr.partial", lambda event: partials.append(event))

    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0),
        config=SessionConfig(
            language=Language.JA,
            partial_interval_ms=100.0,
            vad=VADConfig(silence_to_end_ms=2_000.0),
        ),
    )

    await session.start()
    await session.push_audio(quiet(300))
    for _ in range(6):
        await session.push_audio(tone(200))
        await _settle()

    assert partials, "expected interim hypotheses while speech was in progress"
    assert all(p.session_id == session.session_id for p in partials)
    await session.finish()


async def test_a_failed_partial_does_not_break_the_session() -> None:
    """A failed interim pass is cosmetic; the final pass covers the same audio."""
    ctx = Context()
    session = StreamingSession(
        ctx,
        MockASR(always_fail=True),
        config=SessionConfig(language=Language.JA, partial_interval_ms=50.0),
    )

    await session.start()
    await session.push_audio(quiet(200))
    await session.push_audio(tone(600))
    await _settle()
    transcript = await session.finish()

    assert transcript is not None  # no exception escaped


async def test_finish_flushes_a_trailing_utterance() -> None:
    """Whatever was said just before hanging up must still be transcribed."""
    ctx = Context()
    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0),
        config=SessionConfig(
            language=Language.JA,
            emit_partials=False,
            vad=VADConfig(silence_to_end_ms=10_000.0),  # would never endpoint
        ),
    )

    await session.start()
    await session.push_audio(quiet(200))
    await session.push_audio(tone(800))
    transcript = await session.finish()

    assert transcript.final_segments, "trailing speech was lost"


async def test_audio_is_released_after_each_utterance() -> None:
    """A session that retains everything grows without bound."""
    ctx = Context()
    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0),
        config=SessionConfig(
            language=Language.JA,
            emit_partials=False,
            vad=VADConfig(silence_to_end_ms=300.0),
        ),
    )

    await session.start()
    await session.push_audio(quiet(300))
    await session.push_audio(tone(600))
    await session.push_audio(quiet(500))
    retained_after_first = len(session._retained)

    await session.push_audio(tone(600))
    await session.push_audio(quiet(500))
    retained_after_second = len(session._retained)

    await session.finish()

    # retention is bounded by the utterance, not the session
    assert retained_after_second <= retained_after_first * 2


async def test_usage_accumulates_across_utterances() -> None:
    ctx = Context()
    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0, cost_per_audio_minute_usd=0.006),
        config=SessionConfig(
            language=Language.JA,
            emit_partials=False,
            vad=VADConfig(silence_to_end_ms=300.0),
        ),
    )

    await session.start()
    await session.push_audio(quiet(300))
    await session.push_audio(tone(600) + quiet(500))
    await session.push_audio(tone(600) + quiet(500))
    await session.finish()

    assert session.usage.audio_seconds > 0
    assert session.usage.cost_usd > 0


async def test_pushing_to_a_closed_session_is_an_error() -> None:
    ctx = Context()
    session = StreamingSession(ctx, MockASR(), config=SessionConfig(emit_partials=False))
    await session.start()
    await session.finish()

    with pytest.raises(RuntimeError, match="closed"):
        await session.push_audio(tone(100))


async def test_finish_is_idempotent() -> None:
    ctx = Context()
    session = StreamingSession(ctx, MockASR(), config=SessionConfig(emit_partials=False))
    await session.start()
    first = await session.finish()
    second = await session.finish()
    assert first.duration == second.duration


async def _settle() -> None:
    """Let scheduled partial-recognition tasks run."""
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)


def test_audio_format_round_trip_for_test_helpers() -> None:
    assert STANDARD_FORMAT.duration_of(len(tone(1000))) == pytest.approx(1.0)
