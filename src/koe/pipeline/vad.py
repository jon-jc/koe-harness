"""Voice activity detection and endpointing.

Endpointing -- deciding when a speaker has finished -- sets the floor on
perceived latency in a voice product. Wait too long and every utterance feels
sluggish; cut too early and you truncate someone mid-sentence and the ASR
mis-recognizes the fragment. It is a latency/accuracy trade with no free
option, so it is exposed as configuration rather than buried in a constant.

The detector is energy-based with an **adaptive noise floor**. A fixed
threshold works in a quiet room and fails in every real one: an office at 45 dB
and a café at 65 dB need different absolute cut-offs, and a laptop's automatic
gain control moves the floor around during a call. Tracking the floor from the
quietest recent frames means the threshold follows the room.

Two asymmetries in the configuration are deliberate:

* **Speech onset triggers faster than speech offset.** Missing the first
  syllable of an utterance costs a word; ending an utterance early costs the
  rest of the sentence. So onset needs less evidence than offset.
* **Japanese needs a longer silence timeout than English.** Sentence-final
  particles and politeness endings (〜ですね, 〜ますので) mean Japanese speakers
  pause mid-sentence in places an English-tuned endpointer reads as "done".
  Cutting there truncates the verb, which in Japanese carries the negation and
  the tense.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field
from enum import StrEnum

from koe.domain.audio import AudioChunk, AudioFormat, Encoding
from koe.text.script import Language


class SpeechState(StrEnum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True, slots=True)
class VADConfig:
    """Endpointing behaviour.

    `silence_to_end_ms` is the single most user-visible number in the realtime
    path: it is added to every utterance's latency, and shortening it truncates
    speech.
    """

    frame_ms: float = 20.0
    #: dB above the tracked noise floor for a frame to count as speech.
    speech_threshold_db: float = 9.0
    #: Consecutive speech frames required to declare speech has started.
    onset_frames: int = 2
    #: Silence after speech before an utterance is considered finished.
    silence_to_end_ms: float = 700.0
    #: Force a cut on very long utterances so a monologue still yields results.
    max_utterance_ms: float = 25_000.0
    #: Ignore blips shorter than this -- a cough should not open an utterance.
    min_utterance_ms: float = 200.0
    #: How fast the noise floor adapts (0-1 per frame; higher is faster).
    noise_adaptation: float = 0.05

    @classmethod
    def for_language(cls, language: Language) -> VADConfig:
        """Endpointing tuned per language.

        Japanese gets a longer silence window: speakers pause before
        sentence-final particles and politeness endings, and an English-tuned
        endpointer cuts there -- removing the verb, which is where Japanese
        carries negation and tense.
        """
        if language in (Language.JA, Language.MIXED):
            return cls(silence_to_end_ms=900.0)
        return cls(silence_to_end_ms=650.0)


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """A detected span of speech."""

    start: float
    end: float
    reason: str = "endpoint"

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_energy_db(samples: array.array[int]) -> float:
    """RMS energy of a frame in dBFS.

    Silence returns -100 rather than -inf so the noise-floor tracker stays
    finite; an infinity here propagates into every threshold comparison.
    """
    if not samples:
        return -100.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    rms = math.sqrt(total / len(samples))
    if rms <= 0:
        return -100.0
    return 20.0 * math.log10(rms / 32768.0)


@dataclass(slots=True)
class VAD:
    """Streaming voice activity detector with endpointing.

    Push audio; get back completed :class:`SpeechSegment` values. Stateful and
    owned by exactly one session.
    """

    config: VADConfig = field(default_factory=VADConfig)
    format: AudioFormat = field(default_factory=AudioFormat)

    _state: SpeechState = field(default=SpeechState.SILENCE, init=False)
    _noise_floor_db: float = field(default=-60.0, init=False)
    _consecutive_speech: int = field(default=0, init=False)
    _silence_ms: float = field(default=0.0, init=False)
    _speech_start: float = field(default=0.0, init=False)
    _position: float = field(default=0.0, init=False)
    _pending: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)

    @property
    def state(self) -> SpeechState:
        return self._state

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    @property
    def in_speech(self) -> bool:
        return self._state is SpeechState.SPEECH

    def _frame_bytes(self) -> int:
        return self.format.bytes_for(self.config.frame_ms / 1000.0)

    def push(self, chunk: AudioChunk | bytes) -> list[SpeechSegment]:
        """Feed audio; return any utterances that ended in this chunk."""
        if self.format.encoding is not Encoding.PCM_S16LE:
            raise ValueError(f"VAD needs PCM16, got {self.format.encoding}")

        payload = chunk.data if isinstance(chunk, AudioChunk) else chunk
        self._pending.extend(payload)

        frame_bytes = self._frame_bytes()
        segments: list[SpeechSegment] = []
        if frame_bytes <= 0:
            return segments

        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            segment = self._consume_frame(frame)
            if segment is not None:
                segments.append(segment)
        return segments

    def _consume_frame(self, frame: bytes) -> SpeechSegment | None:
        samples = array.array("h")
        samples.frombytes(frame)
        energy_db = frame_energy_db(samples)

        if not self._initialized:
            # Seed from the first frame instead of a constant, so a loud room
            # does not spend its first second reporting continuous speech.
            self._noise_floor_db = energy_db
            self._initialized = True

        is_speech = energy_db > self._noise_floor_db + self.config.speech_threshold_db

        # Adapt only on non-speech, or the floor climbs to meet the speaker and
        # the detector goes deaf partway through a sentence.
        if not is_speech:
            alpha = self.config.noise_adaptation
            self._noise_floor_db = (1 - alpha) * self._noise_floor_db + alpha * energy_db

        frame_ms = self.config.frame_ms
        self._position += frame_ms / 1000.0
        segment: SpeechSegment | None = None

        if self._state is SpeechState.SILENCE:
            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.config.onset_frames:
                    self._state = SpeechState.SPEECH
                    # Back-date the start so the onset frames are not clipped.
                    self._speech_start = max(
                        0.0,
                        self._position - (self._consecutive_speech * frame_ms) / 1000.0,
                    )
                    self._silence_ms = 0.0
            else:
                self._consecutive_speech = 0
        else:
            if is_speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += frame_ms
                if self._silence_ms >= self.config.silence_to_end_ms:
                    segment = self._close("endpoint")

            if segment is None and self._state is SpeechState.SPEECH:
                speaking_ms = (self._position - self._speech_start) * 1000.0
                if speaking_ms >= self.config.max_utterance_ms:
                    segment = self._close("max-duration")

        return segment

    def _close(self, reason: str) -> SpeechSegment | None:
        end = self._position - (self._silence_ms / 1000.0 if reason == "endpoint" else 0.0)
        start = self._speech_start
        self._state = SpeechState.SILENCE
        self._consecutive_speech = 0
        self._silence_ms = 0.0

        if (end - start) * 1000.0 < self.config.min_utterance_ms:
            return None  # a cough, a door, a keyboard
        return SpeechSegment(start=start, end=end, reason=reason)

    def flush(self) -> SpeechSegment | None:
        """Close any open utterance at end of stream.

        Without this the last thing a caller said is lost whenever they stop
        talking and immediately hang up, which is most calls.
        """
        if self._state is not SpeechState.SPEECH:
            return None
        return self._close("flush")

    def reset(self) -> None:
        self._state = SpeechState.SILENCE
        self._consecutive_speech = 0
        self._silence_ms = 0.0
        self._speech_start = 0.0
        self._position = 0.0
        self._pending.clear()
        self._initialized = False
