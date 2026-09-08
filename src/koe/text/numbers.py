"""Japanese numeral conversion (漢数字 → アラビア数字).

ASR systems are inconsistent about numbers. The same utterance can come back as
``二千二十五年``, ``2025年``, or ``二〇二五年`` depending on the backend and its
inverse-text-normalization settings. Scoring those against each other without
normalizing first produces error rates that measure formatting rather than
recognition -- so this module folds every spelling to one.

Japanese writes numbers two ways and both appear in ASR output:

* **positional** (直読み) -- ``二〇二五`` = 2025, digits in sequence with 〇 as zero
* **additive** (位取り) -- ``二千二十五`` = 2025, digits multiplied by place words

Both are handled, as are 大字 (壱弐参...), the formal forms that show up in
financial and legal contexts.

**The hard part is knowing what is a number at all.** ``一般`` (general),
``一緒`` (together) and ``十分`` (sufficient) all begin with numeral kanji and
none of them are numbers. Converting them yields ``1般``, which is worse than
doing nothing. There is no way to resolve this from characters alone -- it
needs morphology. So :func:`to_arabic` accepts a `gate` callback: when MeCab is
installed the normalizer passes one backed by part-of-speech tags (convert only
tokens tagged 名詞-数詞), and without it we fall back to a conservative
stoplist of the common lexicalized forms. The fallback is deliberately
cautious: a missed conversion costs a little consistency, a wrong one corrupts
the text.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Digit characters, including 大字 (formal forms used on contracts and invoices)
# and the full-width Arabic digits that survive un-normalized input.
_DIGITS: dict[str, int] = {
    "〇": 0, "零": 0, "０": 0,
    "一": 1, "壱": 1, "１": 1,
    "二": 2, "弐": 2, "２": 2,
    "三": 3, "参": 3, "３": 3,
    "四": 4, "肆": 4, "４": 4,
    "五": 5, "伍": 5, "５": 5,
    "六": 6, "陸": 6, "６": 6,
    "七": 7, "漆": 7, "７": 7,
    "八": 8, "捌": 8, "８": 8,
    "九": 9, "玖": 9, "９": 9,
}  # fmt: skip

_SMALL_UNITS: dict[str, int] = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_LARGE_UNITS: dict[str, int] = {"万": 10**4, "億": 10**8, "兆": 10**12, "京": 10**16}

_NUMERAL_CHARS = frozenset(_DIGITS) | frozenset(_SMALL_UNITS) | frozenset(_LARGE_UNITS)
_NUMERAL_RUN = re.compile(f"[{''.join(sorted(_NUMERAL_CHARS))}]+")

#: Lexicalized words that begin with numeral kanji but are not numbers.
#: Used only when morphological analysis is unavailable.
LEXICAL_STOPLIST: frozenset[str] = frozenset(
    {
        "一般", "一緒", "一部", "一方", "一体", "一応", "一切", "一気", "一時",
        "一定", "一致", "一同", "一層", "一見", "一瞬", "一石二鳥", "一番",
        "二度", "三角", "四角", "四苦八苦", "五感", "十分", "十字", "千差万別",
        "万一", "万能", "百科", "百貨", "千葉", "八百屋", "四国", "九州",
    }
)  # fmt: skip


def parse_kanji_number(text: str) -> int | None:
    """Parse a run of numeral characters into an integer.

    Returns ``None`` if `text` is not a well-formed numeral.

    >>> parse_kanji_number("二千二十五")
    2025
    >>> parse_kanji_number("二〇二五")
    2025
    >>> parse_kanji_number("一億二千万")
    120000000
    """
    if not text:
        return None

    # Positional style: a bare digit sequence with no place words. 〇 only ever
    # appears in this style, and a multi-digit run without units must be it.
    if all(ch in _DIGITS for ch in text):
        if len(text) == 1:
            return _DIGITS[text]
        return int("".join(str(_DIGITS[ch]) for ch in text))

    total = 0  # accumulated across 万/億/兆 groups
    section = 0  # accumulated within the current group
    current = 0  # digits awaiting a place word
    saw_value = False

    for ch in text:
        if ch in _DIGITS:
            current = current * 10 + _DIGITS[ch]
            saw_value = True
        elif ch in _SMALL_UNITS:
            # a bare 十 means 10, not 0 -- 十五 is fifteen
            current = current or 1
            section += current * _SMALL_UNITS[ch]
            current = 0
            saw_value = True
        elif ch in _LARGE_UNITS:
            section += current
            current = 0
            # a bare 万 means 10000
            total += (section or 1) * _LARGE_UNITS[ch]
            section = 0
            saw_value = True
        else:
            return None

    return total + section + current if saw_value else None


def _default_gate(run: str, before: str, after: str) -> bool:
    """Conservative heuristic used when morphology is unavailable."""
    # Never convert inside a known lexicalized word.
    for word in LEXICAL_STOPLIST:
        if run and (run + after[: len(word) - len(run)]).startswith(word):
            return False
        if word.startswith(run) and (run + after).startswith(word):
            return False
    # 々 repeats the previous character; a numeral before it is not a count.
    return not after.startswith("々")


def to_arabic(
    text: str,
    *,
    gate: Callable[[str, str, str], bool] | None = None,
) -> str:
    """Rewrite numeral runs in `text` as Arabic digits.

    `gate` receives ``(run, text_before, text_after)`` and returns whether the
    run should be converted. Supply a morphology-backed gate for correctness;
    the default is the conservative stoplist heuristic.

    >>> to_arabic("会議は二千二十五年三月十日です")
    '会議は2025年3月10日です'
    >>> to_arabic("一般的な話")
    '一般的な話'
    """
    decide = gate or _default_gate

    def replace(match: re.Match[str]) -> str:
        run = match.group(0)
        before = text[: match.start()]
        after = text[match.end() :]
        if not decide(run, before, after):
            return run
        value = parse_kanji_number(run)
        return run if value is None else str(value)

    return _NUMERAL_RUN.sub(replace, text)


def contains_numeral(text: str) -> bool:
    """Whether `text` contains any Japanese numeral character."""
    return any(ch in _NUMERAL_CHARS for ch in text)
