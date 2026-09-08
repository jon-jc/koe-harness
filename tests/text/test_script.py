"""Script analysis and JA/EN language identification."""

from __future__ import annotations

import pytest

from koe.text.script import (
    Language,
    Script,
    classify,
    code_switch_ratio,
    detect_language,
    primary_language,
    profile,
    spans,
)


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("あ", Script.HIRAGANA),
        ("カ", Script.KATAKANA),
        ("ｶ", Script.KATAKANA),  # half-width, still katakana
        ("漢", Script.KANJI),
        ("a", Script.LATIN),
        ("Ａ", Script.LATIN),  # full-width latin
        ("1", Script.DIGIT),
        ("１", Script.DIGIT),
        ("。", Script.PUNCT),
        ("、", Script.PUNCT),
        (" ", Script.SPACE),
        ("　", Script.SPACE),  # ideographic space
    ],
)
def test_classify(char: str, expected: Script) -> None:
    assert classify(char) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("これは会議です", Language.JA),
        ("本日の議事録", Language.JA),
        ("this is a meeting", Language.EN),
        ("Let's review the roadmap", Language.EN),
        ("", Language.UNKNOWN),
        ("123 456", Language.UNKNOWN),  # digits carry no language evidence
    ],
)
def test_detect_language_monolingual(text: str, expected: Language) -> None:
    assert detect_language(text) is expected


def test_a_stray_loanword_does_not_make_text_mixed() -> None:
    """Kana is decisive: one hiragana outweighs a lot of Latin."""
    assert detect_language("そのAPIを確認してください") is Language.JA


def test_substantial_code_switching_is_reported_as_mixed() -> None:
    """Real JA/EN switching in business speech should be visible, not hidden."""
    assert detect_language("Q3のroadmapをreviewしましょう") is Language.MIXED


def test_primary_language_never_returns_mixed() -> None:
    """Callers picking a tokenizer need a definite answer."""
    assert primary_language("Q3のroadmapをreviewしましょう") is Language.JA
    assert primary_language("this is english") is Language.EN


def test_kanji_only_text_resolves_to_japanese() -> None:
    assert detect_language("議事録作成") is Language.JA


def test_profile_reports_composition() -> None:
    prof = profile("KPIを確認")
    assert prof.latin_chars == 3
    assert prof.japanese_chars == 3
    assert prof.has_kana
    assert prof.japanese_ratio == pytest.approx(0.5)


def test_spans_split_on_script_family() -> None:
    result = [(s.text, s.is_japanese) for s in spans("そのKPIを確認")]
    assert result == [("その", True), ("KPI", False), ("を確認", True)]


def test_spans_keep_kanji_and_kana_together() -> None:
    """Splitting 食べる into 食 + べる would fragment every verb in the language."""
    result = [s.text for s in spans("食べる")]
    assert result == ["食べる"]


def test_spans_on_empty_text() -> None:
    assert list(spans("")) == []


def test_code_switch_ratio() -> None:
    assert code_switch_ratio("これは日本語です") == pytest.approx(0.0)
    assert code_switch_ratio("all english here") == pytest.approx(0.0)
    assert code_switch_ratio("KPIを確認") == pytest.approx(0.5)
