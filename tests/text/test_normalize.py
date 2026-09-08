"""Normalization: the scoring profile vs the display profile."""

from __future__ import annotations

import pytest

from koe.text.normalize import (
    NormalizationConfig,
    Normalizer,
    normalize_for_display,
    normalize_for_scoring,
)
from koe.text.script import Language

# --------------------------------------------------------------------------
# width and encoding artifacts
# --------------------------------------------------------------------------


def test_fullwidth_latin_folds_to_halfwidth() -> None:
    """Vendors disagree on width and the difference is never meaningful."""
    assert normalize_for_scoring("ＫＰＩ") == "kpi"
    assert normalize_for_display("ＫＰＩ") == "KPI"


def test_halfwidth_katakana_folds_to_fullwidth() -> None:
    assert normalize_for_display("ｶﾀｶﾅ") == "カタカナ"


def test_prolonged_sound_mark_variants_unify() -> None:
    assert normalize_for_display("デ─タ") == "データ"
    assert normalize_for_display("あーーーー") == "あー"


# --------------------------------------------------------------------------
# the whitespace rule
# --------------------------------------------------------------------------


def test_japanese_spaces_are_removed() -> None:
    """Japanese has no word spaces, so any the ASR inserted are arbitrary."""
    assert normalize_for_scoring("これ は 会議 です") == "これは会議です"


def test_english_spaces_are_preserved() -> None:
    """A naive strip would turn 'hello world' into 'helloworld'."""
    assert normalize_for_scoring("hello world") == "hello world"


def test_mixed_text_keeps_english_spacing_and_drops_japanese_spacing() -> None:
    assert normalize_for_scoring("その KPI を review します") == "そのkpiをreviewします"


def test_ideographic_space_is_handled() -> None:
    assert normalize_for_scoring("会議　開始") == "会議開始"


# --------------------------------------------------------------------------
# scoring vs display
# --------------------------------------------------------------------------


def test_scoring_profile_strips_punctuation() -> None:
    assert normalize_for_scoring("本日は、会議です。") == "本日は会議です"


def test_display_profile_preserves_punctuation_and_case() -> None:
    """Stripping 。and 、 would make user-visible Japanese unreadable."""
    assert normalize_for_display("本日は、会議です。") == "本日は、会議です。"
    assert normalize_for_display("The Q3 Roadmap") == "The Q3 Roadmap"


def test_scoring_folds_case() -> None:
    assert normalize_for_scoring("The Q3 Roadmap") == "the q3 roadmap"


def test_same_content_different_formatting_scores_identically() -> None:
    """The whole point: formatting must not show up in the error rate."""
    a = normalize_for_scoring("本日は、KPIを確認します。")
    b = normalize_for_scoring("本日は ＫＰＩ を 確認します")
    assert a == b


# --------------------------------------------------------------------------
# fillers
# --------------------------------------------------------------------------


def test_japanese_fillers_are_removed_for_scoring() -> None:
    """Disfluency policy is a transcription convention, not a recognition error."""
    assert normalize_for_scoring("えーと、会議を始めます") == "会議を始めます"
    assert normalize_for_scoring("あのー、そうですね") == "そうですね"


def test_english_fillers_are_removed_for_scoring() -> None:
    assert normalize_for_scoring("um, let's start the meeting") == "let's start the meeting"


def test_display_profile_keeps_fillers() -> None:
    assert "えーと" in normalize_for_display("えーと、会議を始めます")


# --------------------------------------------------------------------------
# numerals
# --------------------------------------------------------------------------


def test_numeral_spellings_converge_under_scoring() -> None:
    """A model writing 2025年 must not lose to one writing 二〇二五年."""
    assert normalize_for_scoring("二千二十五年") == normalize_for_scoring("2025年")


def test_display_profile_does_not_rewrite_numerals() -> None:
    assert normalize_for_display("二千二十五年") == "二千二十五年"


def test_lexicalized_numerals_survive_normalization() -> None:
    assert "一般" in normalize_for_scoring("一般的な話です")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_config_can_be_customized() -> None:
    cfg = NormalizationConfig.for_scoring().with_(strip_punctuation=False)
    assert Normalizer(cfg).normalize("会議、開始") == "会議、開始"


def test_explicit_language_overrides_detection() -> None:
    # forcing EN skips Japanese numeral conversion
    norm = Normalizer(NormalizationConfig.for_scoring())
    assert norm.normalize("二千二十五", Language.EN) == "二千二十五"


def test_empty_input() -> None:
    assert normalize_for_scoring("") == ""
    assert normalize_for_display("") == ""


@pytest.mark.parametrize("text", ["会議", "meeting", "KPIを確認", "  ", "。、"])
def test_normalization_is_idempotent(text: str) -> None:
    """Running it twice must not change the result -- evals apply it repeatedly."""
    once = normalize_for_scoring(text)
    assert normalize_for_scoring(once) == once
