"""The detector benchmark: eight conditions a meeting room actually produces.

Shipped rather than kept in a scratch directory, because the numbers in the
:mod:`koe.pipeline.vad` docstring and in the README are claims, and a claim
whose measurement nobody else can run is an assertion. ``koe vad-bench``
reproduces them.

**The control column is not optional.** A benchmark reported only as a delta
hides the case where the baseline itself was broken -- and that happened here:
an early version of the unvoiced-run rule fired even with voicing disabled,
where nothing ever resets it, so the "before" column cut every utterance over
400 ms and the improvement looked twice as large as it was. Always read the
control.
"""

from __future__ import annotations

from koe.evaluation.acoustics import Scene
from koe.evaluation.vad import Span, VADReport, score_vad
from koe.pipeline.vad import VAD, VADConfig


def scenes() -> dict[str, Scene]:
    """The conditions, each chosen for a distinct way of being hard."""
    out: dict[str, Scene] = {}

    # A good microphone in a quiet room: the case that already worked, kept so
    # a change that helps the hard conditions cannot quietly ruin the easy one.
    out["quiet room"] = (
        Scene(12.0, bed_level_db=-64.0)
        .speak(1.0, 2.4, -26.0, seed=11)
        .speak(4.6, 1.8, -26.0, seed=12)
        .speak(8.0, 3.0, -26.0, seed=13)
    )

    # A laptop with its fan running: speech about 13 dB over the room.
    out["fan noise"] = (
        Scene(12.0, bed_level_db=-47.0)
        .under("fan", -49.0)
        .speak(1.0, 2.4, -34.0, seed=11)
        .speak(4.6, 1.8, -34.0, seed=12)
        .speak(8.0, 3.0, -34.0, seed=13)
    )

    # The far side of a meeting table: about 7 dB over the room, which is the
    # speaker an over-cautious threshold stops hearing entirely.
    out["far speaker"] = (
        Scene(12.0, bed_level_db=-60.0)
        .speak(1.0, 2.4, -53.0, seed=11)
        .speak(4.6, 1.8, -53.0, seed=12)
        .speak(8.0, 3.0, -53.0, seed=13)
    )

    # A door and a desk bump between sentences.
    out["thumps"] = (
        Scene(14.0, bed_level_db=-62.0)
        .speak(1.0, 2.4, -26.0, seed=11)
        .distract("thump", 4.0, 0.5, -22.0)
        .speak(5.6, 1.8, -26.0, seed=12)
        .distract("thump", 8.2, 0.5, -20.0)
        .distract("thump", 9.4, 0.4, -24.0)
        .speak(10.6, 2.4, -26.0, seed=13)
    )

    # Someone taking notes throughout: the merge case.
    out["keyboard"] = (
        Scene(14.0, bed_level_db=-62.0)
        .speak(1.0, 2.4, -26.0, seed=11)
        .distract("keyboard", 4.2, 2.4, -30.0)
        .speak(7.2, 1.8, -26.0, seed=12)
        .distract("keyboard", 9.6, 1.8, -30.0)
        .speak(11.6, 2.0, -26.0, seed=13)
    )

    # Paper, or a sleeve across the microphone.
    out["rustling"] = (
        Scene(13.0, bed_level_db=-62.0)
        .speak(1.0, 2.4, -26.0, seed=11)
        .distract("rustle", 4.4, 1.6, -30.0)
        .speak(6.6, 2.2, -26.0, seed=12)
        .distract("rustle", 9.4, 1.4, -28.0)
    )

    # Mains hum under everything: periodic, but nowhere near a pitch, and the
    # case that fools a naive periodicity measure.
    out["mains hum"] = (
        Scene(12.0, bed_level_db=-60.0)
        .under("hum", -44.0)
        .speak(1.0, 2.4, -28.0, seed=11)
        .speak(4.6, 1.8, -28.0, seed=12)
        .speak(8.0, 3.0, -28.0, seed=13)
    )

    # Nobody speaks at all. Everything reported here is a false alarm, and this
    # is the condition an energy detector cannot pass.
    out["no speech"] = (
        Scene(14.0, bed_level_db=-60.0)
        .distract("thump", 2.0, 0.5, -22.0)
        .distract("keyboard", 4.0, 3.0, -28.0)
        .distract("rustle", 8.0, 2.0, -28.0)
        .distract("thump", 11.5, 0.5, -20.0)
    )
    return out


def detect(pcm: bytes, config: VADConfig) -> list[Span]:
    """Every span one detector finds in one scene."""
    vad = VAD(config=config)
    spans = [Span(segment.start, segment.end) for segment in vad.push(pcm)]
    trailing = vad.flush()
    if trailing is not None:
        spans.append(Span(trailing.start, trailing.end))
    return spans


def run(config: VADConfig, built: dict[str, Scene] | None = None) -> VADReport:
    """Score `config` across every condition."""
    report = VADReport()
    for name, scene in (built or scenes()).items():
        report.add(
            name,
            score_vad(scene.speech, detect(scene.pcm(), config), audio_seconds=scene.duration_s),
        )
    return report
