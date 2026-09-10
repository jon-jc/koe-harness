"""Cheap acoustic features, for telling speech from everything else that is loud.

An energy detector cannot distinguish a sentence from a door closing. Both are
a rise above the room, both last long enough to clear a minimum duration, and
an endpointer built on energy alone will open an utterance for either, send it
to a recognizer, and get back a confident transcription of a door. In a meeting
room the loud non-speech events are constant -- a laptop lid, a chair, a
keyboard, paper, a cough -- and every one of them costs a recognition request
and a line of nonsense in the transcript.

What separates them is structure, not level.

**Voiced speech is periodic.** Vowels are a glottal pulse train through a
filter, so they repeat at 70-350 Hz. A thud, a click and a hiss do not repeat.
Periodicity is the single strongest speech cue available for the price, and
:func:`periodicity` measures it as a normalized autocorrelation peak over the
plausible pitch range.

**Voiced speech is bottom-heavy.** Its spectrum falls at roughly 6-12 dB per
octave, so most of the energy sits under 1 kHz. Fricatives and rustling paper
are the opposite, and broadband noise is flat. :func:`tilt_db` measures that
with two one-pole filters.

**Fricatives cross zero constantly.** /s/ and /ʃ/ have a zero-crossing rate
several times that of a vowel, which is what separates them from vowels -- and
from the low rumble that tilt alone would confuse with voicing.

No single feature is sufficient, and the combination is not a classifier: see
:mod:`koe.pipeline.vad` for how they are used, which is as evidence for one
decision per utterance rather than a per-frame verdict.

**On cost.** This is a realtime path with a per-session budget, and pure Python
arithmetic is not free. So the features are ordered by price and computed on
demand: energy and zero-crossing rate are a single pass each, tilt is two
one-pole filters, and periodicity -- the expensive one, at a few hundred
microseconds -- is computed only when a decision actually depends on it. The
VAD asks for it while an utterance is unproven and stops asking once it is,
which is a handful of frames per utterance rather than fifty per second.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

#: Everything here assumes koe's standard rate. The decimation factor and the
#: pitch lag range below are derived from it, so a different rate needs them
#: recomputed rather than merely rescaled.
SAMPLE_RATE = 16_000

#: Periodicity is searched on audio decimated by this factor. 4 kHz keeps the
#: fundamental and its first harmonics -- which is all a periodicity measure
#: needs -- and makes the lag search sixteen times cheaper than at full rate.
DECIMATION = 4
DECIMATED_RATE = SAMPLE_RATE // DECIMATION

#: The pitch range koe searches, in Hz. Wide enough for a low male voice and a
#: high female or child one; deliberately not wider, because every extra hertz
#: at the bottom costs lags in the inner loop and admits more slow rumble.
MIN_F0_HZ = 75.0
MAX_F0_HZ = 350.0

MIN_LAG = int(DECIMATED_RATE / MAX_F0_HZ)
MAX_LAG = int(DECIMATED_RATE / MIN_F0_HZ)

#: Samples compared at each lag. Shorter is cheaper and noisier. 96 samples at
#: 4 kHz is 24 ms, which spans at least two periods of even the lowest pitch
#: searched, so a match means real repetition rather than a coincidence.
CORRELATION_WINDOW = 96

#: Audio kept for the periodicity search: enough for the window plus the
#: longest lag. A 20 ms frame is far too short to see a 75 Hz period, so the
#: search runs over a rolling buffer rather than the frame in isolation.
PERIODICITY_SAMPLES = CORRELATION_WINDOW + MAX_LAG + 1

#: Below this the frame is silence and its shape is noise about noise. Skips
#: the arithmetic and, more importantly, stops a near-silent frame reporting a
#: spurious structure that the decision would then act on.
QUIET_FLOOR_DBFS = -70.0


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame's acoustic shape.

    `periodicity` is -1.0 when it was not computed, which is a third state and
    not a low score: "no evidence either way" must not read as "unvoiced", or
    every frame the VAD chose not to pay for would count against the utterance.
    """

    energy_db: float
    zcr: float = 0.0
    tilt_db: float = 0.0
    periodicity: float = -1.0

    @property
    def measured(self) -> bool:
        return self.periodicity >= 0.0

    @property
    def voiced(self) -> bool:
        """Periodic, bottom-heavy and not crossing zero like a fricative.

        All three, because each alone has a common counterexample: a hum is
        periodic, a desk thump is bottom-heavy, and a vowel and a rustle can
        share a zero-crossing rate. Requiring agreement is what makes the
        answer worth acting on.
        """
        return (
            self.periodicity >= VOICED_PERIODICITY
            and self.tilt_db >= VOICED_TILT_DB
            and self.zcr <= VOICED_MAX_ZCR
        )


#: A normalized autocorrelation peak this high means the waveform repeats.
#: Clean vowels reach 0.9; 0.55 admits the ends of words, where the glottal
#: pulse is weakening and the period is drifting, without admitting noise --
#: which sits near zero and does not creep up as it gets louder.
VOICED_PERIODICITY = 0.55

#: How much more energy a voiced frame carries below 1 kHz than above it.
#: Speech falls at 6-12 dB per octave; broadband noise is flat, so 3 dB
#: separates them with room to spare on both sides.
VOICED_TILT_DB = 3.0

#: Vowels sit near 0.05-0.15 at 16 kHz; fricatives run 0.3 and up. The bar is
#: set high enough that a voiced frame riding on room noise still clears it.
VOICED_MAX_ZCR = 0.25


def energy_db(samples: array.array[int]) -> float:
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


def zero_crossing_rate(samples: array.array[int]) -> float:
    """Sign changes per sample, after removing DC.

    The DC removal is not optional. A microphone with a bias offset produces a
    waveform that never crosses zero at all, and its zero-crossing rate reads
    as 0 -- indistinguishable from the lowest-pitched vowel imaginable, on
    every frame, including the ones that are silence.
    """
    count = len(samples)
    if count < 2:
        return 0.0

    mean = sum(samples) / count
    crossings = 0
    previous = samples[0] - mean
    for index in range(1, count):
        current = samples[index] - mean
        # Zero itself counts as neither sign, so a sample sitting exactly on
        # the axis does not manufacture two crossings out of one.
        if (previous > 0.0 and current < 0.0) or (previous < 0.0 and current > 0.0):
            crossings += 1
        if current != 0.0:
            previous = current
    return crossings / (count - 1)


def tilt_db(samples: array.array[int]) -> float:
    """Energy below roughly 1 kHz minus energy above it, in dB.

    Two one-pole filters rather than an FFT: the question is which half of the
    spectrum holds the energy, and a one-pole answers it for a handful of
    operations per sample where a transform would cost a hundred. The cutoff
    is approximate by construction and does not need to be otherwise.
    """
    if not samples:
        return 0.0

    # One-pole coefficient for a ~1 kHz cutoff at 16 kHz:
    # a = exp(-2*pi*fc/fs) = exp(-2*pi*1000/16000).
    alpha = 0.6753
    low = 0.0
    low_energy = 0.0
    high_energy = 0.0
    for sample in samples:
        value = float(sample)
        low = alpha * low + (1.0 - alpha) * value
        high = value - low
        low_energy += low * low
        high_energy += high * high

    # Both floored well below 16-bit quantization, so digital silence gives 0
    # rather than a ratio of two roundings.
    return 10.0 * math.log10((low_energy + 1e-9) / (high_energy + 1e-9))


def decimate(samples: array.array[int], factor: int = DECIMATION) -> array.array[int]:
    """Average every `factor` samples.

    Averaging rather than dropping: dropping samples aliases everything above
    the new Nyquist straight down into the band the pitch search runs in, which
    is the one place a false periodicity would be most costly. A box average is
    a crude anti-alias filter, and crude is the correct amount of effort for a
    signal about to be reduced to a single correlation score.
    """
    out = array.array("i")
    total = len(samples) - (len(samples) % factor)
    for start in range(0, total, factor):
        accumulated = 0
        for offset in range(factor):
            accumulated += samples[start + offset]
        out.append(accumulated // factor)
    return out


def _highpass(decimated: array.array[int]) -> list[float]:
    """Remove everything below the pitch floor, in place of nothing.

    Mains hum at 50 or 60 Hz and HVAC rumble sit below the lowest pitch koe
    searches, and both are in essentially every office recording. Left in, they
    dominate a correlation that is supposed to be about the voice on top of
    them. One pole is enough: the point is to stop rumble deciding the answer,
    not to remove it cleanly.
    """
    # a = exp(-2*pi*60/4000): a ~60 Hz corner, just under MIN_F0_HZ.
    alpha = 0.9099
    out: list[float] = []
    low = 0.0
    for sample in decimated:
        value = float(sample)
        low = alpha * low + (1.0 - alpha) * value
        out.append(value - low)
    return out


def periodicity(decimated: array.array[int]) -> float:
    """Strength of the strongest genuine repetition in the pitch range, 0..1.

    Normalized on both windows, so the score measures *shape* rather than
    level: a quiet vowel and a loud one score the same, and a signal that is
    merely getting louder does not read as increasingly periodic.

    **The score must come from an interior local maximum, and this is the part
    that matters.** A plain "highest correlation in the range" is not a
    periodicity measure at all -- it is a smoothness measure. Any slowly
    varying signal resembles itself a few milliseconds later, so a 50 Hz hum
    scores 0.71 and a low thud 0.89 on the naive version, and both then read as
    voiced speech. What distinguishes real repetition is that the correlation
    *dips and comes back*: a periodic signal has a trough between period
    multiples and a peak at them, where a smooth one decays monotonically and
    peaks only at the edge of the search. Requiring the peak to have a lower
    neighbour on each side rejects the smooth case outright, and costs one
    comparison per lag.

    Expects at least :data:`PERIODICITY_SAMPLES`; returns 0 below that, since a
    buffer too short to hold two periods of the lowest pitch searched cannot
    answer the question it is being asked.
    """
    count = len(decimated)
    if count < CORRELATION_WINDOW + MIN_LAG + 1:
        return 0.0

    window = CORRELATION_WINDOW
    highest_lag = min(MAX_LAG, count - window - 1)
    if highest_lag < MIN_LAG + 2:
        # Not enough lags to have an interior one, so nothing can be a peak.
        return 0.0

    signal = _highpass(decimated)

    # The most recent audio, which is the frame the caller is asking about
    # rather than whatever preceded it.
    base_start = count - window
    reference = signal[base_start:count]
    reference_energy = 0.0
    for value in reference:
        reference_energy += value * value
    if reference_energy <= 0.0:
        return 0.0

    scores: list[float] = []
    for lag in range(MIN_LAG, highest_lag + 1):
        start = base_start - lag
        if start < 0:
            break
        product = 0.0
        lagged_energy = 0.0
        for index in range(window):
            a = reference[index]
            b = signal[start + index]
            product += a * b
            lagged_energy += b * b
        if lagged_energy <= 0.0:
            scores.append(0.0)
            continue
        scores.append(product / math.sqrt(reference_energy * lagged_energy))

    best = 0.0
    for index in range(1, len(scores) - 1):
        score = scores[index]
        # A strict rise into it and a non-rise out of it: a plateau at the top
        # of a real period peak should still count, a monotonic slope should
        # not.
        if score > scores[index - 1] and score >= scores[index + 1] and score > best:
            best = score
    return max(0.0, min(1.0, best))


def describe(samples: array.array[int], *, energy: float | None = None) -> Frame:
    """Every cheap feature for one frame. Periodicity is *not* included.

    The split is the whole cost story: this is one pass for energy, one for
    zero crossings and one for tilt, and it is affordable on every frame.
    Periodicity needs a rolling buffer and an inner loop, and is asked for
    separately by a caller that has decided the answer will change something.
    """
    level = energy_db(samples) if energy is None else energy
    if level <= QUIET_FLOOR_DBFS:
        # Silence has no shape worth measuring, and measuring it anyway invites
        # a decision built on the arithmetic of rounding error.
        return Frame(energy_db=level)
    return Frame(
        energy_db=level,
        zcr=zero_crossing_rate(samples),
        tilt_db=tilt_db(samples),
    )
