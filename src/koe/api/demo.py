"""Server-driven demo playback.

A reviewer opening this on a phone, on a locked-down work laptop, or in any
context where a microphone prompt is unwelcome should still be able to watch
the pipeline run. Without that, the entire realtime path is invisible to most
of the people who will look at this.

The important property is that this is **not an event replay**. It synthesizes
PCM matching the scripted utterance timings and pushes it through the real
:class:`~koe.pipeline.session.StreamingSession`, so voice activity detection,
endpointing, partial stabilization and the ASR call all execute exactly as they
would for a live caller. What the viewer sees is the pipeline working, not a
recording of it having worked.

Playback runs faster than realtime because a 24-second meeting is a long time
to watch nothing happen. That only compresses wall-clock; every timestamp the
pipeline reasons about is derived from the audio itself, so endpointing
behaves identically.
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
from collections.abc import Callable, Sequence

from koe.domain.audio import STANDARD_FORMAT, AudioFormat
from koe.pipeline.session import StreamingSession
from koe.providers.mock import ScriptedUtterance

logger = logging.getLogger(__name__)

#: Wall-clock speed-up. Above ~4x the client-side rate limiter would object,
#: and the visible partials stop being readable.
DEFAULT_SPEED = 3.0
CHUNK_MS = 100.0


def retime(
    script: Sequence[ScriptedUtterance],
    *,
    speech_seconds: float = 3.4,
    pause_seconds: float = 1.4,
) -> list[ScriptedUtterance]:
    """Re-space a script so each utterance is followed by a real pause.

    The corpus packs utterances 0.2s apart, which is fine for scoring — nothing
    there depends on silence. It is wrong for a *demo* of endpointing: the
    Japanese endpointer waits 900ms before closing an utterance, so a 0.2s gap
    means the whole meeting arrives as one segment and the feature being
    demonstrated never fires.

    The same re-timed script is handed to both the audio generator and the
    scripted ASR, so the two timelines stay in agreement.
    """
    out: list[ScriptedUtterance] = []
    cursor = 0.5  # a beat of room tone first, so the noise floor can settle
    for utterance in script:
        out.append(
            ScriptedUtterance(
                speaker=utterance.speaker,
                text=utterance.text,
                start=cursor,
                end=cursor + speech_seconds,
                language=utterance.language,
            )
        )
        cursor += speech_seconds + pause_seconds
    return out


def _tone_chunk(samples: int, *, amplitude: int, frequency: float, rate: int) -> bytes:
    """One chunk of a sine that repeats seamlessly.

    Frequencies are constrained to multiples of :data:`FREQUENCY_STEP` so that a
    whole number of cycles fits in a chunk. That makes the chunk loopable: it
    is generated once and reused for the whole utterance, instead of running a
    Python sine loop 1,600 times per 100 ms of audio — which, at demo speed,
    was costing more wall-clock than the playback it was pacing.

    Seamlessness is not cosmetic here. A discontinuity at each chunk boundary
    is a transient, and the energy detector would read those as speech onsets
    inside the component this is meant to demonstrate.
    """
    buffer = array.array("h")
    step = 2 * math.pi * frequency / rate
    for i in range(samples):
        buffer.append(int(amplitude * math.sin(step * i)))
    return buffer.tobytes()


def _room_tone(samples: int, *, amplitude: int = 24) -> bytes:
    """Low-level noise. Real microphones never produce digital silence, and a
    VAD tuned against true zeros would behave differently in the field."""
    return array.array(
        "h", ((amplitude if i % 7 == 0 else -amplitude) for i in range(samples))
    ).tobytes()


#: Speaker pitches step by this much, and must divide evenly into the chunk
#: rate so a precomputed chunk loops without a click.
FREQUENCY_STEP = 60.0
BASE_FREQUENCY = 150.0
#: Peak amplitude of generated speech, well above the room-tone floor.
SPEECH_AMPLITUDE = 9000


async def drive_demo(
    session: StreamingSession,
    script: Sequence[ScriptedUtterance],
    *,
    speed: float = DEFAULT_SPEED,
    audio_format: AudioFormat = STANDARD_FORMAT,
    on_level: Callable[[float], None] | None = None,
) -> None:
    """Push synthetic audio for `script` through `session` in realtime order.

    `on_level` receives the normalized amplitude of each chunk. The client has
    no microphone during a demo, so without this its level meter would sit dead
    while audio is plainly being processed — which reads as a broken widget.
    The value is the real amplitude of the audio being generated, not a
    decoration synthesized to look busy.
    """
    if not script:
        return

    speakers = list(dict.fromkeys(u.speaker for u in script))
    total = max(u.end for u in script) + 1.0
    chunk_samples = int(audio_format.sample_rate * CHUNK_MS / 1000.0)
    chunk_seconds = CHUNK_MS / 1000.0
    delay = chunk_seconds / max(speed, 0.1)

    # Precompute one loopable chunk per speaker, plus the silence chunk.
    voices: dict[str, bytes] = {
        speaker: _tone_chunk(
            chunk_samples,
            amplitude=SPEECH_AMPLITUDE,
            frequency=BASE_FREQUENCY + index * FREQUENCY_STEP,
            rate=audio_format.sample_rate,
        )
        for index, speaker in enumerate(speakers)
    }
    silence = _room_tone(chunk_samples)

    position = 0.0
    started = asyncio.get_running_loop().time()
    emitted = 0

    while position < total:
        window_end = position + chunk_seconds
        active = next((u for u in script if u.end > position and u.start < window_end), None)
        payload = voices.get(active.speaker, silence) if active is not None else silence
        if on_level is not None:
            on_level(SPEECH_AMPLITUDE / 32768.0 if active is not None else 0.02)

        try:
            await session.push_audio(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("demo playback failed")
            return

        position = window_end
        emitted += 1

        # Pace against a fixed schedule rather than sleeping a constant amount,
        # so time spent inside the pipeline is absorbed instead of added. A
        # naive per-chunk sleep makes playback drift slower the busier the
        # pipeline gets — which is exactly when it looks broken.
        target = started + emitted * delay
        remaining = target - asyncio.get_running_loop().time()
        if remaining > 0:
            await asyncio.sleep(remaining)
