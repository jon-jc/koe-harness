"""Realtime pipeline: endpointing, stabilization, and session orchestration."""

from __future__ import annotations

import array
import math
import random

import pytest

from koe.domain.audio import STANDARD_FORMAT
from koe.kernel.context import Context
from koe.pipeline.session import PartialEvent, SessionConfig, StreamingSession
from koe.pipeline.stabilizer import Stabilizer, common_prefix
from koe.pipeline.vad import MIN_RELEASE_DB, VAD, SpeechState, VADConfig, frame_energy_db
from koe.pipeline.vocabulary import VocabularyStore
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


def _pcm(values: list[float], level_dbfs: float) -> bytes:
    """Scale a float waveform so its RMS lands on `level_dbfs`, as PCM16.

    Normalizing by measured RMS rather than by peak is what makes the dB in
    these tests mean the same thing the detector measures. Scaling by peak
    instead leaves a waveform-shaped offset between the number in the test and
    the number in the threshold comparison, which is how a test ends up
    asserting a signal-to-noise ratio it is not actually producing.
    """
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if rms <= 0:
        return array.array("h", [0] * len(values)).tobytes()
    gain = (10 ** (level_dbfs / 20.0)) * 32768.0 / rms
    return array.array(
        "h", (max(-32768, min(32767, int(value * gain))) for value in values)
    ).tobytes()


def voice(ms: float, level_dbfs: float) -> bytes:
    """Speech-like audio at `level_dbfs`.

    A pure tone is too easy: its energy is perfectly flat, so a detector that
    falls apart on the envelope of real speech passes anyway. This carries the
    syllable-rate amplitude modulation that is what actually breaks
    single-threshold detectors.
    """
    count = int(SAMPLE_RATE * ms / 1000.0)
    values = []
    phase = 0.0
    for index in range(count):
        phase += 2 * math.pi * 150.0 / SAMPLE_RATE
        value = (math.sin(phase) + 0.5 * math.sin(2 * phase)) / 1.5
        values.append(value * (0.7 + 0.3 * math.sin(index / 800.0)))
    return _pcm(values, level_dbfs)


def room(ms: float, level_dbfs: float, *, seed: int = 5) -> bytes:
    """Room tone at `level_dbfs` -- broadband noise, not a repeating pattern."""
    rng = random.Random(seed)
    count = int(SAMPLE_RATE * ms / 1000.0)
    return _pcm([rng.uniform(-1.0, 1.0) for _ in range(count)], level_dbfs)


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
# VAD: the conditions it shipped broken in
#
# Every test above uses a tone 49 dB over its room tone, which is a studio and
# not a customer. These four are the ones that were reported as "the voice
# detection is not working", reproduced at the signal-to-noise ratios real
# hardware actually delivers.
# --------------------------------------------------------------------------


def test_speech_survives_a_room_only_13_db_below_it() -> None:
    """A laptop microphone with a fan running.

    Speech 13 dB over the floor dips between syllables. A single threshold
    reads each dip as silence *and* adapts the floor toward the speaker while
    it does, so the bar rises with every dip until nothing clears it. Measured,
    this returned a two-second utterance 0.76 s long.
    """
    vad = VAD(config=VADConfig(silence_to_end_ms=900.0))
    vad.push(room(1_000, -47.0))
    vad.push(voice(2_000, -34.0))
    segments = vad.push(room(1_600, -47.0))

    assert len(segments) == 1
    assert segments[0].duration == pytest.approx(2.0, abs=0.25)


def test_a_speaker_only_7_db_over_the_room_is_still_heard() -> None:
    """The far side of a meeting table. At the old 9 dB bar: no segments."""
    vad = VAD(config=VADConfig(silence_to_end_ms=700.0))
    vad.push(room(1_000, -60.0))
    vad.push(voice(2_000, -53.0))
    segments = vad.push(room(1_400, -60.0))

    assert len(segments) == 1
    assert segments[0].duration == pytest.approx(2.0, abs=0.3)


def test_the_floor_never_rises_while_an_utterance_is_open() -> None:
    """The feedback loop, asserted directly rather than through its symptom."""
    vad = VAD(config=VADConfig(silence_to_end_ms=5_000.0))
    vad.push(room(400, -50.0))
    vad.push(voice(200, -34.0))
    assert vad.state is SpeechState.SPEECH

    at_onset = vad.noise_floor_db
    vad.push(voice(1_800, -34.0))

    assert vad.state is SpeechState.SPEECH
    assert vad.noise_floor_db <= at_onset


def test_a_leading_frame_of_digital_silence_does_not_deafen_the_detector() -> None:
    """The one that was reported as "the voice detection is not working".

    Captures routinely open with a frame or two of exact zeroes before audio
    starts flowing. Seeding the floor from the first frame put it at the clamp
    -- about 10 dB *below* the actual room -- so room tone itself cleared the
    bar. The detector latched into speech on the first frame and never left, no
    utterance ever ended, and nothing was transcribed. The level meter moved
    the whole time, which is what made it look like a transcription bug.
    """
    vad = VAD(config=VADConfig(silence_to_end_ms=700.0))
    vad.push(bytes(640))  # one 20 ms frame of exact zeroes
    vad.push(room(1_000, -65.0))
    assert vad.state is SpeechState.SILENCE

    vad.push(voice(1_500, -54.0))
    assert vad.state is SpeechState.SPEECH

    segments = vad.push(room(1_200, -65.0))
    assert len(segments) == 1


def test_a_capture_that_opens_mid_word_recovers_at_the_first_pause() -> None:
    """A floor sitting too high is deaf, and every frame it spends coming back
    down is a frame of speech nobody hears. So it comes down fast."""
    vad = VAD(config=VADConfig(silence_to_end_ms=700.0))
    vad.push(voice(400, -30.0))  # the stream opens mid-utterance
    vad.push(room(800, -60.0))  # the speaker pauses
    vad.push(voice(1_500, -45.0))  # and carries on, quieter than before
    segments = vad.push(room(1_200, -60.0))

    assert len(segments) >= 1
    assert segments[-1].duration == pytest.approx(1.5, abs=0.4)


def test_nothing_is_called_speech_before_the_room_is_known() -> None:
    """Committing to a speech call on an unknown floor risks committing to the
    wrong one for the whole session; losing 300 ms is the cheaper trade."""
    vad = VAD()
    assert vad.calibrating

    vad.push(room(100, -60.0))
    assert vad.calibrating
    assert vad.state is SpeechState.SILENCE

    vad.push(room(400, -60.0))
    assert not vad.calibrating


def test_the_floor_stops_before_digital_silence() -> None:
    """Browser noise suppression emits true zeroes between words. A floor that
    follows them to -100 dB puts the onset bar down among the dither."""
    vad = VAD()
    vad.push(bytes(2 * SAMPLE_RATE))  # one second of exact silence

    assert vad.noise_floor_db == pytest.approx(VADConfig().min_noise_floor_db)
    assert vad.state is SpeechState.SILENCE


def test_the_continuation_bar_stays_under_the_onset_bar() -> None:
    """Relative, so that raising the onset bar -- what someone in a loud room
    does -- keeps the hysteresis band the same width instead of turning the
    detector into something that latches on and never lets go."""
    for onset in (3.0, 6.0, 12.0, 24.0):
        config = VADConfig(speech_threshold_db=onset)
        assert MIN_RELEASE_DB <= config.release_threshold_db < onset


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


async def test_the_user_vocabulary_corrects_a_final_segment() -> None:
    """The wiring, end to end: a rule in the store reaches the transcript."""
    ctx = Context()
    store = VocabularyStore()
    store.save("第三四半期 => Q3")
    ctx.provide("vocabulary", store, replace=True)

    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0),
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

    text = "".join(seg.text for seg in session.transcript().final_segments)
    assert "Q3" in text
    assert "第三四半期" not in text


async def test_a_session_without_the_plugin_transcribes_unchanged() -> None:
    """Turning the plugin off gives back the recognizer's own output, not an
    error. The absent service is the pre-existing behaviour."""
    ctx = Context()
    assert ctx.get("vocabulary") is None

    session = StreamingSession(
        ctx,
        MockASR(script=MEETING_JA[:1], degradation=0.0),
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

    text = "".join(seg.text for seg in session.transcript().final_segments)
    assert "第三四半期" in text


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
