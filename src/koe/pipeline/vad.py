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

Everything below is about the floor, because every way this detector has
failed has been a way of getting the floor wrong.

* **The floor is a low quantile of a sliding window, not a running average of
  whatever was not called speech.** The averaging version is a positive
  feedback loop: a speech frame that lands just under the threshold is treated
  as room tone, so the floor moves *up* toward the speaker, which pushes the
  next frame under the threshold too. The detector goes progressively deaf in
  the middle of the sentence feeding it. Measured in a room 13 dB under the
  speaker, it returned a two-second utterance 0.76 s long.

* **It is calibrated over the first 300 ms, and nothing counts as speech until
  it is.** Seeding from a single frame is a coin flip, and losing the flip
  costs the whole session rather than one frame. Captures routinely open with a
  frame or two of exact zeroes before audio flows; seeding from one of those
  put the floor at the clamp, roughly 10 dB *below* the actual room, at which
  point room tone itself cleared the bar. The detector latched into speech on
  the first frame and stayed there, so no utterance ever ended and nothing was
  ever transcribed -- while the level meter moved the whole time, which is what
  made it look like a transcription bug rather than a detection one.

* **A quantile, deliberately not the minimum.** One frame of digital silence
  drags a minimum-based floor to the clamp and reproduces exactly the failure
  above. A tenth-percentile floor discards the outlier and still tracks the
  quiet part of the room, which is the thing being estimated.

* **It falls fast, rises slowly, and never rises while an utterance is open.**
  Falling is safe in a way rising is not: a frame quieter than the floor cannot
  be speech, because speech is what sits above the room, never below it. So a
  capture that opened mid-word recovers at the speaker's first pause instead of
  staying deaf for the rest of the call.

Two more asymmetries have nothing to do with the floor:

* **Speech onset triggers faster than speech offset**, and **the bar to stay in
  speech is lower than the bar to enter it** -- hysteresis. A single threshold
  has to be both sensitive enough to catch a quiet first syllable and steady
  enough to ride out the gap between two words, and no one number is both.

* **Japanese needs a longer silence timeout than English.** Sentence-final
  particles and politeness endings (〜ですね, 〜ますので) mean Japanese speakers
  pause mid-sentence in places an English-tuned endpointer reads as "done".
  Cutting there truncates the verb, which in Japanese carries the negation and
  the tense.
"""

from __future__ import annotations

import array
import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from koe.domain.audio import AudioChunk, AudioFormat, Encoding
from koe.text.script import Language


class SpeechState(StrEnum):
    SILENCE = "silence"
    SPEECH = "speech"


#: The continuation bar is never allowed closer than this to the floor. At zero
#: the detector would hold an utterance open on room tone alone.
MIN_RELEASE_DB = 1.5


@dataclass(frozen=True, slots=True)
class VADConfig:
    """Endpointing behaviour.

    `silence_to_end_ms` is the single most user-visible number in the realtime
    path: it is added to every utterance's latency, and shortening it truncates
    speech.
    """

    frame_ms: float = 20.0
    #: dB above the tracked noise floor for a frame to *open* an utterance.
    #: 6 rather than a rounder, safer-looking 9: a laptop microphone across a
    #: meeting table puts the far side of the room only about 7 dB over its own
    #: noise, and at 9 that speaker is not detected at all -- not truncated,
    #: not late, simply never heard. The cost of the lower bar is paid by
    #: `onset_frames` and `min_utterance_ms`, which is where blips belong.
    speech_threshold_db: float = 6.0
    #: How far *under* the onset bar the continuation bar sits. Relative rather
    #: than absolute, so that raising the onset bar -- what someone in a loud
    #: room does -- keeps the band the same width instead of turning the
    #: detector into something that latches on and never lets go.
    release_margin_db: float = 3.5
    #: Consecutive speech frames required to declare speech has started.
    onset_frames: int = 2
    #: Silence after speech before an utterance is considered finished.
    silence_to_end_ms: float = 700.0
    #: Force a cut on very long utterances so a monologue still yields results.
    max_utterance_ms: float = 25_000.0
    #: Ignore blips shorter than this -- a cough should not open an utterance.
    min_utterance_ms: float = 200.0
    #: Audio observed before the detector will call anything speech. Short
    #: enough to be over before anyone has finished clicking the microphone
    #: button, long enough that the floor is an estimate rather than a guess.
    calibration_ms: float = 300.0
    #: Sliding window the floor estimate is drawn from.
    floor_window_ms: float = 1_500.0
    #: Which quantile of that window counts as "the room".
    floor_quantile: float = 0.1
    #: How fast the floor rises toward a louder room (0-1 per frame).
    noise_adaptation: float = 0.05
    #: How fast it falls toward a quieter one. Much faster on purpose: a floor
    #: sitting too high is deaf, and every frame it spends coming back down is
    #: a frame of speech nobody hears.
    noise_decay: float = 0.30
    #: The floor never goes below this. Browser noise suppression emits
    #: near-digital silence between words, and a floor that follows it all the
    #: way down to -100 dB puts the onset bar among the dither, where the
    #: detector triggers on nothing at all.
    min_noise_floor_db: float = -75.0

    @property
    def release_threshold_db(self) -> float:
        """dB above the floor for a frame to *keep* an open utterance open."""
        return max(self.speech_threshold_db - self.release_margin_db, MIN_RELEASE_DB)

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
    _recent: deque[float] = field(default_factory=deque, init=False, repr=False)
    _frames_seen: int = field(default=0, init=False)

    @property
    def state(self) -> SpeechState:
        return self._state

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    @property
    def in_speech(self) -> bool:
        return self._state is SpeechState.SPEECH

    @property
    def calibrating(self) -> bool:
        """Whether the floor is still being estimated.

        Exposed because a UI that shows a speech indicator should be able to
        say "getting the room" rather than "silence" for the first fraction of
        a second, which is otherwise indistinguishable from not working.
        """
        return self._frames_seen * self.config.frame_ms < self.config.calibration_ms

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

        config = self.config
        frame_ms = config.frame_ms
        self._observe(energy_db)

        if self.calibrating:
            # Nothing counts as speech until the room is known. Committing to a
            # speech/silence call on an unknown floor risks committing to the
            # wrong one for the rest of the session, which is a far worse
            # trade than losing the first fraction of a second.
            self._noise_floor_db = max(self._room_estimate(), config.min_noise_floor_db)
            self._position += frame_ms / 1000.0
            return None

        # Hysteresis: a higher bar to open an utterance than to keep one open.
        # Which bar applies is decided by the state we are *in*, before this
        # frame gets to change it.
        margin = (
            config.release_threshold_db
            if self._state is SpeechState.SPEECH
            else config.speech_threshold_db
        )
        is_speech = energy_db > self._noise_floor_db + margin

        self._adapt_floor(is_speech)

        self._position += frame_ms / 1000.0
        segment: SpeechSegment | None = None

        if self._state is SpeechState.SILENCE:
            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= config.onset_frames:
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
                if self._silence_ms >= config.silence_to_end_ms:
                    segment = self._close("endpoint")

            if segment is None and self._state is SpeechState.SPEECH:
                speaking_ms = (self._position - self._speech_start) * 1000.0
                if speaking_ms >= config.max_utterance_ms:
                    segment = self._close("max-duration")

        return segment

    def _observe(self, energy_db: float) -> None:
        """Record one frame's energy in the sliding window."""
        window = max(1, int(self.config.floor_window_ms / self.config.frame_ms))
        self._recent.append(energy_db)
        while len(self._recent) > window:
            self._recent.popleft()
        self._frames_seen += 1

    def _room_estimate(self) -> float:
        """The low quantile of the window: what the room has sounded like.

        Independent of the speech decision, which is the point. A floor derived
        from "frames we did not call speech" is derived from the threshold it
        then feeds, and closes a loop that ends with the detector deaf.
        """
        if not self._recent:
            return self._noise_floor_db
        ordered = sorted(self._recent)
        index = min(len(ordered) - 1, int(len(ordered) * self.config.floor_quantile))
        return ordered[index]

    def _adapt_floor(self, is_speech: bool) -> None:
        """Move the floor toward the room estimate, asymmetrically.

        Downward is always safe: the estimate dropping means the room got
        quieter, and speech cannot be quieter than the room it is in. Upward is
        only safe between utterances -- raising the floor while one is open is
        how the detector used to talk itself deaf mid-sentence.
        """
        config = self.config
        candidate = self._room_estimate()

        if candidate < self._noise_floor_db:
            alpha = config.noise_decay
        elif self._state is SpeechState.SILENCE and not is_speech:
            alpha = config.noise_adaptation
        else:
            return

        adapted = (1 - alpha) * self._noise_floor_db + alpha * candidate
        self._noise_floor_db = max(adapted, config.min_noise_floor_db)

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
        self._recent.clear()
        self._frames_seen = 0
