"""Writing-script analysis for Japanese/English text.

Language identification for JA/EN does not need a statistical model. The two
languages use disjoint scripts, so counting codepoints is both faster and more
reliable than an n-gram classifier -- and, unlike a model, it degrades in ways
you can reason about.

The part that does need care is **code-switching**. Japanese business speech
mixes Latin script constantly::

    そのKPIを来週までにアップデートします
    Q3のロードマップをレビューしましょう

A naive "majority script wins" rule labels these Japanese and moves on, which
is right for choosing a tokenizer but wrong for normalization: the Latin spans
need English rules (case folding) while the Japanese spans need Japanese ones
(width folding, numeral conversion). So this module reports the *composition*
of a string, not just a winner, and exposes span segmentation so the
normalizer can treat each run on its own terms.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum


class Script(StrEnum):
    """A writing system, at the granularity language ID needs."""

    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    KANJI = "kanji"
    LATIN = "latin"
    DIGIT = "digit"
    PUNCT = "punct"
    SPACE = "space"
    OTHER = "other"

    @property
    def is_japanese(self) -> bool:
        return self in (Script.HIRAGANA, Script.KATAKANA, Script.KANJI)


class Language(StrEnum):
    """The languages koe supports, plus the mixed and unknown cases."""

    JA = "ja"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# Codepoint ranges. Half-width katakana is included because ASR vendors and
# older Japanese systems still emit it, and it must fold to full-width before
# anything is compared.
_HIRAGANA = ((0x3041, 0x309F),)
_KATAKANA = ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))
_KANJI = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))
_JA_PUNCT = frozenset("、。「」『』（）〜・！？：；，．…‥〽“”‘’　【】〔〕《》〈〉")


def _in(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def classify(char: str) -> Script:
    """Classify a single character by writing system."""
    cp = ord(char)
    if char.isspace() or char == "　":
        return Script.SPACE
    if _in(cp, _HIRAGANA):
        return Script.HIRAGANA
    if _in(cp, _KATAKANA):
        # the prolonged sound mark belongs to whichever kana run it sits in;
        # katakana is the overwhelmingly common case
        return Script.KATAKANA
    if _in(cp, _KANJI):
        return Script.KANJI
    if char in _JA_PUNCT:
        return Script.PUNCT
    if char.isdigit():
        return Script.DIGIT
    if char.isalpha() and cp < 0x2000:
        return Script.LATIN
    # full-width latin (ＡＢＣ) normalizes to latin
    if 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return Script.LATIN
    if 0xFF10 <= cp <= 0xFF19:
        return Script.DIGIT
    if unicodedata.category(char).startswith("P"):
        return Script.PUNCT
    return Script.OTHER


@dataclass(frozen=True, slots=True)
class ScriptProfile:
    """The script composition of a string."""

    counts: dict[Script, int]
    total: int

    @property
    def japanese_chars(self) -> int:
        return sum(self.counts.get(s, 0) for s in (Script.HIRAGANA, Script.KATAKANA, Script.KANJI))

    @property
    def latin_chars(self) -> int:
        return self.counts.get(Script.LATIN, 0)

    @property
    def scored_chars(self) -> int:
        """Characters that carry language evidence (excludes space/punct/digits)."""
        return self.japanese_chars + self.latin_chars

    @property
    def japanese_ratio(self) -> float:
        return self.japanese_chars / self.scored_chars if self.scored_chars else 0.0

    @property
    def has_kana(self) -> bool:
        """Kana is decisive: it appears in Japanese and nowhere else."""
        return bool(self.counts.get(Script.HIRAGANA, 0) or self.counts.get(Script.KATAKANA, 0))


def profile(text: str) -> ScriptProfile:
    """Count the script composition of `text`."""
    counts: dict[Script, int] = {}
    for ch in text:
        script = classify(ch)
        counts[script] = counts.get(script, 0) + 1
    return ScriptProfile(counts=counts, total=len(text))


def detect_language(text: str, *, mixed_threshold: float = 0.30) -> Language:
    """Identify the language of `text` from its scripts.

    `mixed_threshold` is the share of language-bearing characters the *minority*
    script must reach before the text counts as genuinely code-switched. The
    default separates the two cases that actually occur in Japanese business
    speech:

    * ``そのAPIを確認してください`` -- one borrowed acronym inside a Japanese
      sentence. Minority share ~0.21, so: Japanese.
    * ``Q3のroadmapをreviewしましょう`` -- clause-level switching. Minority
      share ~0.33, so: mixed.

    The distinction earns its keep because the two want different handling: the
    first is summarized as Japanese, while the second needs each language's
    normalization rules applied to its own spans.

    Callers that only need to choose a tokenizer should use
    :func:`primary_language`, which never returns ``MIXED``.
    """
    prof = profile(text)
    if prof.scored_chars == 0:
        return Language.UNKNOWN
    if prof.japanese_chars == 0:
        return Language.EN

    ja_ratio = prof.japanese_ratio
    minority = min(ja_ratio, 1.0 - ja_ratio)

    if minority >= mixed_threshold:
        return Language.MIXED
    # Kanji without kana is still Japanese here: koe supports only JA and EN, so
    # the CJK ambiguity a general-purpose detector would face does not arise.
    return Language.JA if ja_ratio > 0.5 else Language.EN


def primary_language(text: str) -> Language:
    """Language ID collapsed to JA or EN, for choosing a tokenizer or prompt."""
    lang = detect_language(text)
    if lang is Language.MIXED:
        return Language.JA if profile(text).has_kana else Language.EN
    return lang


@dataclass(frozen=True, slots=True)
class Span:
    """A maximal run of one script family."""

    text: str
    script: Script
    start: int
    end: int

    @property
    def is_japanese(self) -> bool:
        return self.script.is_japanese


def spans(text: str) -> Iterator[Span]:
    """Split `text` into maximal runs of the same script family.

    Kana and kanji are merged into one Japanese family, because a Japanese word
    routinely spans both (``食べる`` is kanji + hiragana) and splitting there
    would fragment every verb in the language.
    """

    def family(s: Script) -> str:
        if s.is_japanese:
            return "ja"
        if s is Script.LATIN:
            return "latin"
        return "neutral"

    if not text:
        return

    start = 0
    current_script = classify(text[0])
    current_family = family(current_script)

    for i, ch in enumerate(text[1:], start=1):
        script = classify(ch)
        fam = family(script)
        if fam != current_family:
            yield Span(text[start:i], current_script, start, i)
            start = i
            current_script = script
            current_family = fam
        elif current_script in (Script.SPACE, Script.PUNCT) and script.is_japanese:
            current_script = script

    yield Span(text[start:], current_script, start, len(text))


def code_switch_ratio(text: str) -> float:
    """Fraction of language-bearing characters in the minority script.

    A useful operational signal: a spike in code-switching on a session often
    means technical vocabulary the ASR model is about to get wrong.
    """
    prof = profile(text)
    if prof.scored_chars == 0:
        return 0.0
    ja = prof.japanese_ratio
    return min(ja, 1.0 - ja)
