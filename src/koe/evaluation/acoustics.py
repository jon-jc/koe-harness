"""Synthetic audio with known ground truth, for scoring a detector.

A VAD benchmark needs audio where the speech spans are known exactly. Recorded
audio has to be hand-labelled, and hand labels disagree about boundaries by
more than the differences being measured. Synthetic audio has the labels for
free, and -- more usefully -- lets a specific failure be constructed on
purpose: a door closing at 4.2 s, a keyboard between two sentences, a speaker
7 dB over the room.

**These signals are not speech and the numbers they produce are not accuracy
figures.** They are a harness for the *decision*: given a signal with speech's
structure and a signal with furniture's, does the detector separate them, and
where does it put the boundaries. A detector that scores badly here is
certainly broken; one that scores well still has to be tried on a microphone,
which is why this module is a benchmark and not a substitute for one.

The distractors are chosen because they are what actually sets off an
energy-based detector in a meeting room, and each has a different structure:

* a **door or desk thump** -- loud, low, brief, not periodic
* a **keyboard** -- broadband clicks, high zero-crossing rate
* **paper or fabric** -- sustained high-frequency hiss
* a **fan or air conditioner** -- steady, which the noise floor should absorb
* **mains hum** -- periodic but far below any pitch, and the case that fools a
  naive periodicity measure
"""

from __future__ import annotations

import array
import math
import random
from dataclasses import dataclass, field

from koe.evaluation.vad import Span

SAMPLE_RATE = 16_000


def _pcm(values: list[float], level_dbfs: float) -> array.array[int]:
    """Scale a float waveform so its RMS lands on `level_dbfs`, as PCM16."""
    if not values:
        return array.array("h")
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if rms <= 0:
        return array.array("h", [0] * len(values))
    gain = (10 ** (level_dbfs / 20.0)) * 32768.0 / rms
    return array.array("h", (max(-32768, min(32767, int(value * gain))) for value in values))


def voiced(samples: int, *, f0: float = 130.0, start_phase: float = 0.0) -> list[float]:
    """A glottal pulse train through formants: a vowel, structurally.

    Harmonics falling at roughly 12 dB per octave with resonances near 700 and
    1200 Hz. What matters is not that it sounds like a person -- it does not --
    but that it is *periodic and bottom-heavy*, which is what voicing is and
    what any speech/non-speech decision has to key on.
    """
    out = []
    for index in range(samples):
        t = index / SAMPLE_RATE + start_phase
        value = 0.0
        harmonic = 1
        while harmonic * f0 < 5_000 and harmonic < 30:
            frequency = f0 * harmonic
            amplitude = 1.0 / (harmonic**1.6)
            if 500 < frequency < 900:
                amplitude *= 3.0
            elif 1_000 < frequency < 1_500:
                amplitude *= 2.0
            value += amplitude * math.sin(2 * math.pi * frequency * t)
            harmonic += 1
        out.append(value)
    return out


def unvoiced(samples: int, *, seed: int = 7) -> list[float]:
    """A fricative: high-passed noise, as in /s/ or /ʃ/."""
    rng = random.Random(seed)
    out: list[float] = []
    previous = 0.0
    for _ in range(samples):
        value = rng.uniform(-1.0, 1.0)
        out.append(value - 0.85 * previous)
        previous = value
    return out


def utterance(duration_s: float, *, f0: float = 130.0, seed: int = 11) -> list[float]:
    """Syllables: voiced runs with fricatives and short closures between them.

    The closures matter more than they look. Real speech is not a continuous
    tone, and a detector that only ever sees one will not have to cope with the
    gap between two words -- which is where a single-threshold endpointer cuts
    a sentence in half.
    """
    rng = random.Random(seed)
    total = int(duration_s * SAMPLE_RATE)
    out: list[float] = []
    phase = 0.0
    while len(out) < total:
        # A syllable: optional onset fricative, then a voiced nucleus.
        if rng.random() < 0.35:
            length = int(rng.uniform(0.04, 0.09) * SAMPLE_RATE)
            out.extend(value * 0.5 for value in unvoiced(length, seed=rng.randrange(10_000)))
        length = int(rng.uniform(0.10, 0.22) * SAMPLE_RATE)
        pitch = f0 * rng.uniform(0.9, 1.15)
        nucleus = voiced(length, f0=pitch, start_phase=phase)
        phase += length / SAMPLE_RATE
        # Fade the edges so syllables do not click into each other.
        fade = max(1, int(0.005 * SAMPLE_RATE))
        for index in range(fade):
            nucleus[index] *= index / fade
            nucleus[-1 - index] *= index / fade
        out.extend(nucleus)
        # An inter-syllable closure: quiet, not silent.
        gap = int(rng.uniform(0.01, 0.06) * SAMPLE_RATE)
        out.extend(0.02 * rng.uniform(-1.0, 1.0) for _ in range(gap))
    return out[:total]


def room_tone(samples: int, *, seed: int = 3) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(samples)]


def fan(samples: int, *, seed: int = 5) -> list[float]:
    """Air conditioning: low-passed noise, steady."""
    rng = random.Random(seed)
    out: list[float] = []
    low = 0.0
    for _ in range(samples):
        low = 0.97 * low + 0.03 * rng.uniform(-1.0, 1.0)
        out.append(low)
    return out


def hum(samples: int, *, frequency: float = 50.0) -> list[float]:
    return [math.sin(2 * math.pi * frequency * i / SAMPLE_RATE) for i in range(samples)]


def thump(samples: int) -> list[float]:
    """A door or a desk: a low transient that decays fast."""
    out = []
    for index in range(samples):
        t = index / SAMPLE_RATE
        envelope = math.exp(-t * 22.0)
        out.append(
            envelope * (math.sin(2 * math.pi * 68 * t) + 0.4 * math.sin(2 * math.pi * 150 * t))
        )
    return out


def keyboard(samples: int, *, seed: int = 13) -> list[float]:
    """Typing: broadband clicks at a plausible rate."""
    rng = random.Random(seed)
    out = [0.0] * samples
    position = 0
    while position < samples:
        length = min(samples - position, int(0.012 * SAMPLE_RATE))
        for index in range(length):
            envelope = math.exp(-(index / SAMPLE_RATE) * 500)
            out[position + index] = envelope * rng.uniform(-1.0, 1.0)
        position += int(rng.uniform(0.09, 0.19) * SAMPLE_RATE)
    return out


def rustle(samples: int, *, seed: int = 17) -> list[float]:
    """Paper or a sleeve on a microphone: sustained high-frequency noise."""
    rng = random.Random(seed)
    out: list[float] = []
    previous = 0.0
    for _ in range(samples):
        value = rng.uniform(-1.0, 1.0)
        out.append(value - 0.7 * previous)
        previous = value
    return out


@dataclass(slots=True)
class Scene:
    """A stretch of audio being assembled, with its speech spans recorded.

    The ground truth is produced by the same call that produces the audio, so
    the labels cannot drift out of step with the signal they describe -- which
    is the failure mode of every benchmark whose annotations live in a separate
    file.
    """

    duration_s: float
    bed_level_db: float = -62.0
    bed_seed: int = 3
    _mix: list[float] = field(default_factory=list, init=False, repr=False)
    _speech: list[Span] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        samples = int(self.duration_s * SAMPLE_RATE)
        bed = _pcm(room_tone(samples, seed=self.bed_seed), self.bed_level_db)
        self._mix = [float(value) for value in bed]

    def _add(self, values: list[float], at_s: float, level_db: float) -> None:
        scaled = _pcm(values, level_db)
        start = int(at_s * SAMPLE_RATE)
        for index, value in enumerate(scaled):
            position = start + index
            if 0 <= position < len(self._mix):
                self._mix[position] += float(value)

    def speak(self, at_s: float, duration_s: float, level_db: float, *, seed: int = 11) -> Scene:
        """Add an utterance and record it as ground truth."""
        self._add(utterance(duration_s, seed=seed), at_s, level_db)
        self._speech.append(Span(start=at_s, end=at_s + duration_s))
        return self

    def distract(self, kind: str, at_s: float, duration_s: float, level_db: float) -> Scene:
        """Add a non-speech event. Deliberately *not* recorded as speech."""
        samples = int(duration_s * SAMPLE_RATE)
        generators = {
            "thump": lambda: thump(samples),
            "keyboard": lambda: keyboard(samples),
            "rustle": lambda: rustle(samples),
            "fan": lambda: fan(samples),
            "hum": lambda: hum(samples),
        }
        self._add(generators[kind](), at_s, level_db)
        return self

    def under(self, kind: str, level_db: float) -> Scene:
        """Lay a continuous noise under the whole scene."""
        return self.distract(kind, 0.0, self.duration_s, level_db)

    @property
    def speech(self) -> list[Span]:
        return sorted(self._speech, key=lambda span: span.start)

    def pcm(self) -> bytes:
        """The mixed scene as PCM16.

        Clipped rather than rescaled: rescaling after mixing would change every
        signal-to-noise ratio the scene was built to specify.
        """
        return array.array(
            "h", (max(-32768, min(32767, int(value))) for value in self._mix)
        ).tobytes()
