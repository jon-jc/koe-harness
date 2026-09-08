"""Japanese numeral parsing (漢数字)."""

from __future__ import annotations

import pytest

from koe.text.numbers import contains_numeral, parse_kanji_number, to_arabic


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # single digits
        ("一", 1),
        ("九", 9),
        ("〇", 0),
        ("零", 0),
        # small place words
        ("十", 10),
        ("十五", 15),
        ("二十", 20),
        ("二十五", 25),
        ("百", 100),
        ("百二十三", 123),
        ("千", 1000),
        ("三千", 3000),
        # additive style
        ("二千二十五", 2025),
        ("千九百八十四", 1984),
        ("五千四百三十二", 5432),
        # large units
        ("一万", 10_000),
        ("十万", 100_000),
        ("三万五千", 35_000),
        ("一億", 100_000_000),
        ("一億二千万", 120_000_000),
        ("一兆", 1_000_000_000_000),
        # positional style (直読み), how years are usually spoken
        ("二〇二五", 2025),
        ("一九八四", 1984),
        ("二〇二五年", None),  # trailing counter is not part of the numeral run
    ],
)
def test_parse_kanji_number(text: str, expected: int | None) -> None:
    assert parse_kanji_number(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("壱", 1),
        ("弐", 2),
        ("参", 3),
        ("壱萬", None),  # 萬 is not in the supported unit set
        ("参百", 300),
    ],
)
def test_daiji_formal_numerals(text: str, expected: int | None) -> None:
    """大字 appear on invoices and contracts, which meetings discuss."""
    assert parse_kanji_number(text) == expected


def test_parse_rejects_non_numerals() -> None:
    assert parse_kanji_number("") is None
    assert parse_kanji_number("会議") is None
    assert parse_kanji_number("二会") is None


def test_bare_large_unit_means_one_of_it() -> None:
    assert parse_kanji_number("万") == 10_000
    assert parse_kanji_number("億") == 100_000_000


def test_to_arabic_rewrites_numbers_in_context() -> None:
    assert to_arabic("会議は二千二十五年三月十日です") == "会議は2025年3月10日です"
    assert to_arabic("売上は一億二千万円でした") == "売上は120000000円でした"


def test_to_arabic_leaves_lexicalized_words_alone() -> None:
    """一般 must not become 1般 -- a wrong conversion is worse than none."""
    for word in ("一般的な話", "一緒に行く", "一方で", "十分な時間", "千葉支社"):
        assert to_arabic(word) == word


def test_gate_callback_overrides_the_default_heuristic() -> None:
    # a gate that refuses everything leaves the text untouched
    assert to_arabic("二千二十五年", gate=lambda *_: False) == "二千二十五年"
    # and one that accepts everything converts even a stoplisted word
    assert to_arabic("一般", gate=lambda *_: True) == "1般"


def test_contains_numeral() -> None:
    assert contains_numeral("三月")
    assert not contains_numeral("会議")
    assert not contains_numeral("meeting")
