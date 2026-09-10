"""The voicing gate: what it rejects, what it must never reject.

Written against whole scenes rather than frames, because the behaviour being
tested is about an utterance -- whether it is one, and where it ends -- and
neither question exists at frame granularity.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from koe.evaluation import acoustics
from koe.evaluation.acoustics import Scene
from koe.evaluation.vad import Span, score_vad
from koe.pipeline.vad import VAD, VADConfig

VOICED = VADConfig(require_voicing=True)
ENERGY_ONLY = VADConfig(require_voicing=False)


def detect(scene: Scene, config: VADConfig = VOICED) -> list[Span]:
    vad = VAD(config=config)
    spans = [Span(s.start, s.end) for s in vad.push(scene.pcm())]
    trailing = vad.flush()
    if trailing is not None:
        spans.append(Span(trailing.start, trailing.end))
    return spans


# --------------------------------------------------------------------------
# what it rejects
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "level"),
    [("thump", -20.0), ("keyboard", -28.0), ("rustle", -28.0)],
)
def test_a_room_with_no_speech_in_it_produces_no_utterances(kind: str, level: float) -> None:
    """Every one of these is loud enough to open an utterance on energy alone,
    and every one of them returns a confident transcription of furniture."""
    scene = Scene(8.0, bed_level_db=-60.0).distract(kind, 2.0, 2.0, level)
    assert detect(scene) == []


def test_energy_alone_does_not_manage_that() -> None:
    """The control. Without this the test above proves only that the scene is
    quiet, which would be a test of the generator rather than the detector."""
    scene = Scene(8.0, bed_level_db=-60.0).distract("thump", 2.0, 2.0, -20.0)
    assert detect(scene, ENERGY_ONLY) != []


def test_a_door_between_two_sentences_does_not_become_a_third() -> None:
    scene = (
        Scene(12.0, bed_level_db=-62.0)
        .speak(1.0, 2.0, -26.0, seed=11)
        .distract("thump", 4.5, 0.5, -20.0)
        .speak(6.5, 2.0, -26.0, seed=12)
    )
    score = score_vad(scene.speech, detect(scene), audio_seconds=scene.duration_s)
    assert score.detected == 2
    assert score.false_alarms == 0


# --------------------------------------------------------------------------
# what it must never reject
# --------------------------------------------------------------------------


def test_speech_is_still_detected_in_every_condition_that_worked_before() -> None:
    """The gate is only worth having if it costs no recall. A detector that
    rejects noise by rejecting quiet speakers has not improved."""
    for bed, level in [(-64.0, -26.0), (-47.0, -34.0), (-60.0, -53.0)]:
        scene = (
            Scene(9.0, bed_level_db=bed)
            .speak(1.0, 2.4, level, seed=11)
            .speak(5.0, 2.0, level, seed=12)
        )
        score = score_vad(scene.speech, detect(scene), audio_seconds=scene.duration_s)
        assert score.detected == 2, f"bed {bed} speech {level}"


def test_a_fricative_onset_does_not_lose_the_word() -> None:
    """Speech is voiced *or* unvoiced, and requiring every frame to be voiced
    would cut /s/ off the front of every word beginning with one. The rule is
    that an utterance must *contain* voicing, not that every frame must."""
    scene = Scene(8.0, bed_level_db=-62.0)
    scene.distract("rustle", 1.0, 0.12, -26.0)  # stands in for /s/
    scene.speak(1.12, 1.8, -26.0, seed=11)
    spans = detect(scene)
    assert len(spans) == 1
    # It reaches back over the fricative rather than starting after it.
    assert spans[0].start <= 1.15


def test_an_utterance_nobody_looked_at_is_not_discarded() -> None:
    """A configuration that never measures must not silently reject
    everything: absence of evidence is not evidence."""
    never_looks = replace(VOICED, voicing_probe_frames=0, voicing_recheck_frames=10**6)
    scene = Scene(8.0, bed_level_db=-62.0).speak(1.0, 2.0, -26.0, seed=11)
    assert detect(scene, never_looks) != []


# --------------------------------------------------------------------------
# merging, which is what the unvoiced-run rule is for
# --------------------------------------------------------------------------


def test_typing_before_a_sentence_does_not_arrive_glued_to_it() -> None:
    """Measured on the benchmark: 3.4 seconds of keyboard came back as the
    front of an utterance, and half of what reached the recognizer was
    furniture."""
    scene = (
        Scene(12.0, bed_level_db=-62.0)
        .distract("keyboard", 1.0, 2.4, -30.0)
        .speak(4.6, 2.4, -26.0, seed=11)
    )
    spans = detect(scene)
    assert len(spans) == 1
    # Starts at the speech, not two seconds earlier at the typing.
    assert spans[0].start > 4.0


def test_the_merge_rule_is_off_when_voicing_is_off() -> None:
    """It depends on the voicing measurement: with the measurement off nothing
    ever resets the run, and every utterance over 400 ms would be cut. That
    mistake briefly made the benchmark's own baseline look far worse than it
    is."""
    scene = Scene(9.0, bed_level_db=-62.0).speak(1.0, 3.0, -26.0, seed=11)
    spans = detect(scene, ENERGY_ONLY)
    assert len(spans) == 1
    assert spans[0].end - spans[0].start > 2.0


def test_a_long_sentence_is_not_cut_by_the_unvoiced_run() -> None:
    """The rule releases loud frames, it does not close utterances. Real
    speech re-voices long before the endpoint window elapses."""
    scene = Scene(12.0, bed_level_db=-62.0).speak(1.0, 6.0, -26.0, seed=11)
    spans = detect(scene)
    assert len(spans) == 1
    assert spans[0].end - spans[0].start > 5.0


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_the_detector_keeps_up_with_realtime_by_a_wide_margin() -> None:
    """The measurement is only affordable because it is sampled. A regression
    to measuring every frame would still pass a correctness test and quietly
    cut how many sessions a box can carry."""
    import time

    scene = Scene(20.0, bed_level_db=-62.0)
    for index in range(6):
        scene.speak(1.0 + index * 3.0, 2.0, -26.0, seed=11 + index)
    pcm = scene.pcm()

    started = time.perf_counter()
    detect_vad = VAD(config=VOICED)
    detect_vad.push(pcm)
    rtf = (time.perf_counter() - started) / scene.duration_s

    assert rtf < 0.05, f"realtime factor {rtf:.4f}"


def _browser_thump(seconds: float) -> list[float]:
    """The exact signal that exposed the bug: a step to full level and an
    exponential decay, as a Web Audio `exponentialRampToValueAtTime` produces.

    Reproduced rather than approximated, because the accident depends on the
    decay: the spurious frame is in the quiet tail, where the level has fallen
    into the room tone and the measurement is about rounding.
    """
    out = []
    for index in range(int(seconds * acoustics.SAMPLE_RATE)):
        t = index / acoustics.SAMPLE_RATE
        envelope = 0.5 * (0.002 ** (min(t, 0.45) / 0.45))
        out.append(envelope * math.sin(2 * math.pi * 68 * t))
    return out


def test_one_accidentally_periodic_frame_does_not_admit_a_door() -> None:
    """Caught in the browser, not in the benchmark.

    A decaying 68 Hz thump is not periodic in the pitch range and almost every
    frame of it scores 0.000 -- except one, and that one frame was enough to
    admit the whole door, because the rule was "contains voicing" and one frame
    contained it.

    Three frames is a vowel. One is a coincidence.
    """
    rng = random.Random(4)
    signal = (
        [0.0] * int(1.2 * acoustics.SAMPLE_RATE)
        + _browser_thump(1.0)
        + [0.0] * int(1.5 * acoustics.SAMPLE_RATE)
    )
    mixed = [value + rng.uniform(-1.0, 1.0) * 0.0008 for value in signal]
    pcm = acoustics._pcm(mixed, -30.0).tobytes()

    def spans(config: VADConfig) -> list[Span]:
        vad = VAD(config=config)
        found = [Span(s.start, s.end) for s in vad.push(pcm)]
        trailing = vad.flush()
        if trailing is not None:
            found.append(Span(trailing.start, trailing.end))
        return found

    # The control first: at a threshold of one, the door gets in. That is what
    # makes this a test of the threshold rather than of the thump.
    assert spans(replace(VOICED, min_voiced_frames=1)) != []
    assert spans(VOICED) == []


def test_real_speech_clears_the_voiced_frame_bar_easily() -> None:
    """The bar has to be high enough to reject an accident and low enough that
    a short quiet utterance still clears it."""
    scene = Scene(8.0, bed_level_db=-60.0).speak(1.0, 0.6, -50.0, seed=11)
    assert detect(scene) != []
