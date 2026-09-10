"""Acoustic features, and the speech/non-speech decision built on them.

The cases that matter are the confusions. Any feature separates a vowel from
silence; the question is whether it separates a vowel from the specific things
that are loud in a meeting room, because those are what an energy detector
sends to a recognizer and gets a confident transcription of furniture back.
"""

from __future__ import annotations

import array
import math
import time

import pytest

from koe.evaluation import acoustics
from koe.pipeline import features
from koe.pipeline.features import Frame, describe, energy_db, periodicity, zero_crossing_rate

RATE = acoustics.SAMPLE_RATE
FRAME = 320
#: Enough audio for one pitch search.
ROLLING = features.PERIODICITY_SAMPLES * features.DECIMATION


def _at(values: list[float], level_db: float) -> array.array[int]:
    return acoustics._pcm(values, level_db)


def _periodicity_of(values: list[float], level_db: float = -26.0) -> float:
    return periodicity(features.decimate(_at(values, level_db)))


def _shape(values: list[float], level_db: float = -26.0) -> Frame:
    samples = _at(values, level_db)
    partial = describe(samples[-FRAME:])
    return Frame(
        energy_db=partial.energy_db,
        zcr=partial.zcr,
        tilt_db=partial.tilt_db,
        periodicity=_periodicity_of(values, level_db),
    )


# --------------------------------------------------------------------------
# energy and zero crossings
# --------------------------------------------------------------------------


def test_energy_of_silence_is_finite() -> None:
    assert energy_db(array.array("h", [0] * FRAME)) == -100.0
    assert energy_db(array.array("h")) == -100.0


def test_a_dc_offset_does_not_read_as_the_lowest_pitch_imaginable() -> None:
    """A microphone with a bias offset produces a waveform that never crosses
    zero, so an unadjusted rate reads 0 on every frame -- including silence."""
    biased = array.array(
        "h", [8000 + int(2000 * math.sin(2 * math.pi * 300 * i / RATE)) for i in range(FRAME)]
    )
    assert zero_crossing_rate(biased) > 0.0


def test_a_fricative_crosses_zero_far_more_often_than_a_vowel() -> None:
    voiced = zero_crossing_rate(_at(acoustics.voiced(FRAME), -26.0))
    unvoiced = zero_crossing_rate(_at(acoustics.unvoiced(FRAME), -26.0))
    assert unvoiced > voiced * 3


# --------------------------------------------------------------------------
# periodicity: the cases a naive version gets wrong
# --------------------------------------------------------------------------


def test_a_vowel_is_periodic() -> None:
    assert _periodicity_of(acoustics.voiced(ROLLING)) > 0.8


def test_noise_is_not() -> None:
    assert _periodicity_of(acoustics.room_tone(ROLLING)) < 0.5
    assert _periodicity_of(acoustics.unvoiced(ROLLING)) < 0.5


def test_mains_hum_is_not_periodicity_however_periodic_it_looks() -> None:
    """The case that makes "highest correlation in the range" a smoothness
    measure rather than a periodicity one.

    A 50 Hz sine resembles itself a few milliseconds later, so a naive peak
    scored it 0.71 -- comfortably "voiced" -- and every office recording has
    one under it. Requiring an interior local maximum rejects it, because a
    smooth signal's correlation decays monotonically and peaks only at the edge
    of the search.
    """
    assert _periodicity_of(acoustics.hum(ROLLING)) < 0.3


def test_a_low_thump_is_not_periodicity_either() -> None:
    """Same failure, different cause: a door is bottom-heavy and smooth, and
    the naive measure scored it 0.89."""
    assert _periodicity_of(acoustics.thump(ROLLING)) < 0.3


def test_periodicity_is_about_shape_not_level() -> None:
    """Normalized on both windows, so a signal merely getting louder does not
    read as increasingly periodic."""
    quiet = _periodicity_of(acoustics.voiced(ROLLING), -50.0)
    loud = _periodicity_of(acoustics.voiced(ROLLING), -14.0)
    assert quiet == pytest.approx(loud, abs=0.1)


def test_too_little_audio_answers_zero_rather_than_guessing() -> None:
    """A buffer too short to hold two periods of the lowest pitch searched
    cannot answer the question it is being asked."""
    assert periodicity(array.array("i", [1, 2, 3])) == 0.0


# --------------------------------------------------------------------------
# the combined verdict
# --------------------------------------------------------------------------


def test_a_vowel_is_voiced() -> None:
    assert _shape(acoustics.voiced(ROLLING)).voiced


@pytest.mark.parametrize(
    "source",
    ["unvoiced", "room_tone", "hum", "thump", "rustle", "keyboard"],
)
def test_the_things_that_fool_an_energy_detector_are_not_voiced(source: str) -> None:
    """Every one of these is loud enough to open an utterance on energy alone."""
    generator = getattr(acoustics, source)
    assert not _shape(generator(ROLLING)).voiced


def test_no_single_feature_decides_it() -> None:
    """Each has a common counterexample, which is why agreement is required:
    a hum is periodic, a thump is bottom-heavy, and a vowel and a rustle can
    share a zero-crossing rate."""
    assert not Frame(energy_db=-26.0, zcr=0.01, tilt_db=20.0, periodicity=0.2).voiced
    assert not Frame(energy_db=-26.0, zcr=0.01, tilt_db=-10.0, periodicity=0.9).voiced
    assert not Frame(energy_db=-26.0, zcr=0.60, tilt_db=20.0, periodicity=0.9).voiced
    assert Frame(energy_db=-26.0, zcr=0.05, tilt_db=12.0, periodicity=0.9).voiced


def test_an_unmeasured_frame_is_not_the_same_as_an_unvoiced_one() -> None:
    """A third state, and it has to be: every frame the VAD chose not to pay
    for would otherwise count as evidence against the utterance."""
    unmeasured = Frame(energy_db=-26.0)
    assert not unmeasured.measured
    assert not unmeasured.voiced
    assert Frame(energy_db=-26.0, periodicity=0.0).measured


def test_silence_is_not_measured_for_shape() -> None:
    """Measuring the shape of near-silence invites a decision built on the
    arithmetic of rounding error."""
    quiet = describe(array.array("h", [0] * FRAME))
    assert quiet.zcr == 0.0
    assert quiet.tilt_db == 0.0


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_the_cheap_features_stay_cheap() -> None:
    """These run on every loud frame of every concurrent session, so a
    regression here is a capacity regression rather than a slow test.

    The bound is generous against CI hardware; it is here to catch an
    accidental O(n^2) rather than to certify a number.
    """
    frame = _at(acoustics.voiced(FRAME), -26.0)
    started = time.perf_counter()
    for _ in range(200):
        describe(frame)
    per_call_ms = (time.perf_counter() - started) / 200 * 1000.0
    # One 20 ms frame's budget is 20 ms; 2 ms leaves a 10x margin.
    assert per_call_ms < 2.0, f"{per_call_ms:.2f} ms per frame"


def test_periodicity_stays_affordable() -> None:
    """The expensive one, which is why the VAD samples it rather than running
    it on every frame."""
    rolling = features.decimate(_at(acoustics.voiced(ROLLING), -26.0))
    started = time.perf_counter()
    for _ in range(100):
        periodicity(rolling)
    per_call_ms = (time.perf_counter() - started) / 100 * 1000.0
    assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per call"
