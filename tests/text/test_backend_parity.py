"""MeCab and the fallback path must agree where it matters.

Japanese tokenization is optional (``pip install 'koe-harness[ja]'``). That
creates a hazard specific to this project: a developer machine with MeCab and a
CI runner without it can silently disagree, so a normalization bug reproduces
in one place and not the other.

These tests pin the contract by forcing each backend explicitly:

* numeral folding must produce the same result either way, because the eval
  layer depends on it to make two spellings of the same number comparable
* the fallback must never corrupt lexicalized words -- the failure mode that
  makes a wrong conversion worse than no conversion
* the two tokenizers must report *different names*, because a Japanese WER is
  only meaningful alongside the segmenter that produced it
"""

from __future__ import annotations

import pytest

from koe.text import normalize as normalize_mod
from koe.text.normalize import NormalizationConfig, Normalizer
from koe.text.tokenize import (
    CharacterTokenizer,
    WhitespaceTokenizer,
    japanese_tokenizer,
    mecab_available,
)

requires_mecab = pytest.mark.skipif(not mecab_available(), reason="MeCab/fugashi not installed")


@pytest.fixture
def without_mecab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the no-MeCab code path regardless of the environment."""
    monkeypatch.setattr(normalize_mod, "mecab_available", lambda: False)


NUMERAL_CASES = [
    ("二千二十五年", "2025年"),
    ("三月十日", "3月10日"),
    ("一億二千万円", "120000000円"),
    ("二〇二五年", "2025年"),
    ("五十人", "50人"),
]


@pytest.mark.parametrize(("source", "expected"), NUMERAL_CASES)
def test_numeral_folding_without_mecab(source: str, expected: str, without_mecab: None) -> None:
    norm = Normalizer(NormalizationConfig.for_display().with_(convert_numerals=True))
    assert norm.normalize(source) == expected


@requires_mecab
@pytest.mark.parametrize(("source", "expected"), NUMERAL_CASES)
def test_numeral_folding_with_mecab(source: str, expected: str) -> None:
    norm = Normalizer(NormalizationConfig.for_display().with_(convert_numerals=True))
    assert norm.normalize(source) == expected


@pytest.mark.parametrize("text", ["一般的な話", "一緒に行きましょう", "十分な時間", "一方で"])
def test_lexicalized_words_survive_without_mecab(text: str, without_mecab: None) -> None:
    """The fallback is deliberately conservative: 一般 must not become 1般."""
    norm = Normalizer(NormalizationConfig.for_display().with_(convert_numerals=True))
    assert norm.normalize(text) == text


@requires_mecab
@pytest.mark.parametrize("text", ["一般的な話", "一緒に行きましょう", "一方で"])
def test_lexicalized_words_survive_with_mecab(text: str) -> None:
    norm = Normalizer(NormalizationConfig.for_display().with_(convert_numerals=True))
    assert norm.normalize(text) == text


def test_scoring_output_is_identical_across_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scoring profile is what evals compare, so it must not drift."""
    samples = [
        "本日は、KPIを確認します。",
        "会議は二千二十五年三月十日です",
        "えーと、その件は来週までに対応します",
        "Q3のroadmapをreviewしましょう",
    ]
    norm = Normalizer(NormalizationConfig.for_scoring())
    with_mecab = [norm.normalize(s) for s in samples]

    monkeypatch.setattr(normalize_mod, "mecab_available", lambda: False)
    without = [norm.normalize(s) for s in samples]

    assert with_mecab == without


def test_tokenizers_report_distinct_names() -> None:
    """A Japanese WER is only comparable alongside its segmenter's name."""
    assert CharacterTokenizer().name == "character"
    assert WhitespaceTokenizer().name == "whitespace"
    assert japanese_tokenizer(prefer_mecab=False).name == "character"


@requires_mecab
def test_mecab_tokenizer_segments_and_tags() -> None:
    tokens = japanese_tokenizer(prefer_mecab=True).tokenize("会議を始めます")
    assert [t.surface for t in tokens][:2] == ["会議", "を"]
    assert any(t.pos for t in tokens), "MeCab should provide part-of-speech tags"


@requires_mecab
def test_mecab_identifies_numerals_by_pos() -> None:
    """POS tagging is what separates the number 一 from the word 一般."""
    tokenizer = japanese_tokenizer(prefer_mecab=True)
    assert any(t.is_numeral for t in tokenizer.tokenize("五十人が参加"))
    assert not any(t.is_numeral for t in tokenizer.tokenize("一般的な話"))


def test_character_tokenizer_drops_whitespace() -> None:
    assert [t.surface for t in CharacterTokenizer().tokenize("会 議")] == ["会", "議"]
