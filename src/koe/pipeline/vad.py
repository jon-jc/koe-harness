"""Voice activity detection and endpointing.

Endpointing -- deciding when a speaker has finished -- sets the floor on
perceived latency in a voice product. Wait too long and every utterance feels
sluggish; cut too early and you truncate someone mid-sentence and the ASR
mis-recognizes the fragment. It is a latency/accuracy trade with no free
option, so it is exposed as configuration rather than buried in a constant.

The detector has two halves, and they answer different questions.

**Energy decides where something is happening**, against an **adaptive noise
floor**. A fixed threshold works in a quiet room and fails in every real one: an
office at 45 dB and a café at 65 dB need different absolute cut-offs, and a
laptop's automatic gain control moves the floor around during a call. Tracking
the floor from the quietest recent frames means the threshold follows the room.

**Structure decides whether that something is a person.** Energy cannot: a door
closing is a rise above the room that lasts long enough to clear any minimum
duration, and an energy detector opens an utterance for it, sends it to a
recognizer, and gets back a confident transcription of a door. In a meeting room
the loud non-speech events are constant -- a laptop lid, a chair, a keyboard,
paper, a cough -- and each one costs a recognition request and a line of nonsense
that reads exactly like something a person said.

So an utterance must **contain voicing** to be emitted. Not every frame: speech
is voiced *or* unvoiced, and requiring every frame to be periodic would cut /s/
off the front of every word that starts with one. The rule is about the
utterance, because "was that speech" is a question asked once, at the end.

Nor is one voiced frame enough to settle it. A decaying thump produced exactly
one frame in its quiet tail that scored periodic, and admitted the whole door.
Three frames is a vowel; one is a coincidence. :mod:`koe.pipeline.features`
measures the evidence, and what it costs is described there.

The same measurement does a second job: **a run of loud frames with no voicing
in it stops holding an utterance open.** This is what keeps a noise from
*merging* with the sentence beside it. A keyboard is loud enough to keep
resetting the endpoint timer, so without it the typing and the sentence after it
arrive as one segment -- measured, 3.4 seconds of keyboard glued to the front of
an utterance, with half of what reached the recognizer being furniture.

Measured across eight conditions in
``koe.evaluation.acoustics``: precision 0.72 to 0.95, false alarms 3.5 per
minute to zero, merged utterances 2 to 0, with recall unchanged at 0.998 and
about 1.3x the CPU.

The rest of this is about the floor, because every way this detector has failed
in production has been a way of getting the floor wrong.

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
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from koe.domain.audio import AudioChunk, AudioFormat, Encoding
from koe.pipeline import features
from koe.pipeline.features import Frame, energy_db
from koe.text.script import Language

#: Kept at its original name and location because it is part of this module's
#: surface; the implementation now lives with the other frame measurements.
frame_energy_db = energy_db


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

    #: Require an utterance to contain voicing before it is emitted. This is
    #: what separates a sentence from a door, and energy cannot do it: see the
    #: module docstring.
    require_voicing: bool = True
    #: Voiced frames an utterance needs before it counts as speech.
    #:
    #: Not one. One is a coincidence: a decaying 68 Hz thump produced exactly
    #: one frame in its quiet tail that scored periodic, and that single frame
    #: was enough to admit the whole door. Three is a vowel -- voicing runs
    #: 100-200 ms, which is five to ten frames -- so a real utterance clears
    #: this without trying and an accident does not.
    min_voiced_frames: int = 3
    #: How many loud frames to look at every frame before backing off. A real
    #: utterance voices within a syllable or two, so the answer normally
    #: arrives well inside this.
    voicing_probe_frames: int = 40
    #: After that, look every Nth frame instead of stopping -- and keep looking
    #: for the rest of the utterance. Sampling rather than abandoning, because
    #: voicing can arrive late: a rustle or a keyboard burst that runs into the
    #: sentence behind it opens the utterance, spends the dense budget on noise,
    #: and the speech that follows would never be measured at all. That cost a
    #: real utterance in the benchmark. Every tenth frame is a fifth of a second
    #: and a twentieth of the price.
    voicing_recheck_frames: int = 10
    #: How long an open utterance may stay loud without any voicing before it
    #: stops being held open.
    #:
    #: This is what stops a noise *merging* with the sentence next to it. A
    #: keyboard is loud enough to keep resetting the endpoint timer, so without
    #: this the typing and the sentence after it become one segment: measured,
    #: 3.4 seconds of keyboard arrived glued to the front of an utterance and
    #: half of what reached the recognizer was furniture.
    #:
    #: 400 ms is chosen to be longer than any fricative -- /s/ runs 50-200 ms --
    #: so the rule cannot cut inside a word. The case it does get wrong is
    #: whispering, which is unvoiced throughout; koe would end a whispered
    #: utterance early. That is a real cost, accepted because whispering into a
    #: meeting recorder is rare and typing beside one is not.
    max_unvoiced_run_ms: float = 400.0

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

    #: Decimated audio for the pitch search. A 20 ms frame is far too short to
    #: see a 75 Hz period, so periodicity runs over a rolling window rather
    #: than the frame in isolation.
    _decimated: deque[int] = field(default_factory=deque, init=False, repr=False)
    #: Voicing evidence for the utterance in progress. Per utterance, not per
    #: frame: the question "was that speech" is asked once, at the end.
    _voiced_frames: int = field(default=0, init=False)
    _probes_spent: int = field(default=0, init=False)
    #: Milliseconds of loud audio since voicing was last seen.
    _unvoiced_run_ms: float = field(default=0.0, init=False)
    #: The most recent frame's measurements, for the UI and for debugging a
    #: session that is deciding something surprising.
    _last: Frame = field(default_factory=lambda: Frame(energy_db=-100.0), init=False)

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
        self._remember_decimated(samples)

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

        # Energy proposes, spectrum disposes. The shape of a frame is only
        # measured when the frame is loud enough to matter, which in a quiet
        # room is a small fraction of them and in a noisy one is the fraction
        # that could actually be an event.
        self._last = features.describe(samples, energy=energy_db) if is_speech else Frame(energy_db)
        if is_speech:
            self._measure_voicing()

        # Time since voicing was last seen, counted over *every* frame of an
        # open utterance rather than only the loud ones. Counting only loud
        # frames looks more careful and does not work: a keyboard is a burst
        # every 140 ms, so the quiet between two clicks reset the run before it
        # could ever reach the limit, and the typing went on holding the
        # utterance open exactly as before.
        #
        # Counting quiet frames too is safe because the rule only ever takes
        # away a loud frame's ability to reset the endpoint timer. During a
        # genuine pause the timer is already running, so tripping the rule
        # changes nothing about when that pause ends the utterance.
        if self._state is SpeechState.SPEECH:
            self._unvoiced_run_ms += frame_ms
        else:
            self._unvoiced_run_ms = 0.0

        if (
            config.require_voicing
            and is_speech
            and self._state is SpeechState.SPEECH
            and self._unvoiced_run_ms >= config.max_unvoiced_run_ms
        ):
            # Still loud, but nothing about it has been speech for longer than
            # any fricative lasts. Releasing it lets the endpoint timer run, so
            # the utterance closes where the speaking stopped instead of being
            # dragged along by whatever is making the noise.
            #
            # Gated on `require_voicing` because it *depends* on the voicing
            # measurement: with the measurement off nothing ever resets the
            # run, and every utterance over 400 ms would be cut. That mistake
            # briefly made the benchmark's own baseline look far worse than it
            # is, which is a good argument for always reading the control
            # column rather than only the delta.
            is_speech = False

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

    def _remember_decimated(self, samples: array.array[int]) -> None:
        """Keep just enough decimated audio for one pitch search."""
        self._decimated.extend(features.decimate(samples))
        while len(self._decimated) > features.PERIODICITY_SAMPLES:
            self._decimated.popleft()

    def _measure_voicing(self) -> None:
        """Look for voicing on this frame, if looking would tell us anything.

        Two questions are being answered by one measurement, which is why the
        sampling schedule is not simply "until we find some".

        *Is this an utterance at all* -- answered once, so the search is dense
        until the first voiced frame and then stops mattering.

        *Is it still an utterance* -- answered continuously, because a sentence
        that has ended into a keyboard is not still a sentence, and the only
        way to know is to keep looking. So after the dense budget, and after
        voicing is found, the measurement keeps running at a tenth of the rate,
        which is cheap enough to leave on for the length of a call.
        """
        config = self.config
        if not config.require_voicing:
            return

        recheck = max(1, config.voicing_recheck_frames)
        proven = self._voiced_frames >= config.min_voiced_frames
        dense = not proven and self._probes_spent < config.voicing_probe_frames
        if not dense and self._frames_seen % recheck:
            return

        self._probes_spent += 1
        score = features.periodicity(array.array("i", self._decimated))
        self._last = Frame(
            energy_db=self._last.energy_db,
            zcr=self._last.zcr,
            tilt_db=self._last.tilt_db,
            periodicity=score,
        )
        if self._last.voiced:
            self._voiced_frames += 1
            self._unvoiced_run_ms = 0.0

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
        voiced = self._voiced_frames
        probed = self._probes_spent

        self._state = SpeechState.SILENCE
        self._consecutive_speech = 0
        self._silence_ms = 0.0
        self._voiced_frames = 0
        self._probes_spent = 0
        self._unvoiced_run_ms = 0.0

        if (end - start) * 1000.0 < self.config.min_utterance_ms:
            return None  # too short to be a sentence

        if self.config.require_voicing and probed and voiced < self.config.min_voiced_frames:
            # Loud, long enough, and never periodic for as long as a vowel
            # lasts: a door, a chair, a keyboard, paper. Emitting it costs a
            # recognition request and
            # returns a confident transcription of furniture -- which is worse
            # than emitting nothing, because it lands in the transcript looking
            # exactly like something a person said.
            #
            # `probed` guards the case where nothing was measured at all, so a
            # configuration that never looks cannot silently discard every
            # utterance.
            return None

        return SpeechSegment(start=start, end=end, reason=reason)

    def flush(self) -> SpeechSegment | None:
        """Close any open utterance at end of stream.

        Without this the last thing a caller said is lost whenever they stop
        talking and immediately hang up, which is most calls.
        """
        if self._state is not SpeechState.SPEECH:
            return None
        return self._close("flush")

    @property
    def last_frame(self) -> Frame:
        """The most recent frame's measurements.

        Exposed so a UI can show *why* a decision went the way it did, and so a
        session that is behaving strangely can be diagnosed from the numbers
        rather than from a level meter.
        """
        return self._last

    def reset(self) -> None:
        self._state = SpeechState.SILENCE
        self._consecutive_speech = 0
        self._silence_ms = 0.0
        self._speech_start = 0.0
        self._position = 0.0
        self._pending.clear()
        self._recent.clear()
        self._decimated.clear()
        self._frames_seen = 0
        self._voiced_frames = 0
        self._probes_spent = 0
        self._unvoiced_run_ms = 0.0
