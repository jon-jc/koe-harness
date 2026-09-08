"""Session settings arriving from a browser.

These became user-facing controls the moment the settings panel got sliders,
which means the values are now attacker-controlled in the ordinary sense: they
come from a client, and the server has to stay sane whatever arrives. The
interesting cases are not "does 700 work" but "what happens on 0, on 10^9, on
a string, and on nothing at all".
"""

from __future__ import annotations

from koe.api.app import _clamp, _vad_overrides
from koe.pipeline.vad import VADConfig
from koe.text.script import Language

# --------------------------------------------------------------------------
# clamping
# --------------------------------------------------------------------------


def test_a_value_in_range_is_kept() -> None:
    assert _clamp(700, 200.0, 3000.0, 900.0) == 700.0


def test_a_value_out_of_range_is_pulled_to_the_edge() -> None:
    assert _clamp(0, 200.0, 3000.0, 900.0) == 200.0
    assert _clamp(10**9, 200.0, 3000.0, 900.0) == 3000.0


def test_a_non_number_falls_back_to_the_default() -> None:
    for junk in ("soon", None, {}, [], object()):
        assert _clamp(junk, 200.0, 3000.0, 900.0) == 900.0


def test_nan_falls_back_rather_than_propagating() -> None:
    """NaN survives min/max unchanged and would poison every comparison after."""
    assert _clamp(float("nan"), 200.0, 3000.0, 900.0) == 900.0


def test_a_numeric_string_is_accepted() -> None:
    """JSON from a form control legitimately arrives as a string."""
    assert _clamp("650", 200.0, 3000.0, 900.0) == 650.0


# --------------------------------------------------------------------------
# endpointing
# --------------------------------------------------------------------------


def test_no_override_keeps_the_language_default() -> None:
    """Japanese needs a longer silence window; nothing may quietly undo that."""
    assert _vad_overrides({}, Language.JA) is None


def test_an_override_starts_from_the_language_default() -> None:
    """Setting one field must not reset the others to the class defaults."""
    config = _vad_overrides({"speech_threshold_db": 12}, Language.JA)

    assert config is not None
    assert config.speech_threshold_db == 12.0
    # Still the Japanese window, not VADConfig's generic 700.
    assert config.silence_to_end_ms == VADConfig.for_language(Language.JA).silence_to_end_ms


def test_both_fields_apply() -> None:
    config = _vad_overrides({"silence_to_end_ms": 1200, "speech_threshold_db": 6}, Language.EN)

    assert config is not None
    assert config.silence_to_end_ms == 1200.0
    assert config.speech_threshold_db == 6.0


def test_absurd_values_are_clamped_not_honoured() -> None:
    """A zero-length silence window ends an utterance on the first quiet frame."""
    config = _vad_overrides({"silence_to_end_ms": 0, "speech_threshold_db": 1000}, Language.EN)

    assert config is not None
    assert config.silence_to_end_ms == 200.0
    assert config.speech_threshold_db == 24.0
