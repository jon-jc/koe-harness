"""Bilingual (JA/EN) text processing: script analysis, normalization, tokenization.

Japanese is not a locale flag in this codebase, it is a design constraint. The
language has no word spaces, writes numbers three different ways, and mixes
Latin script freely in business speech. Each of those breaks a piece of the
standard English-shaped ASR evaluation pipeline, so koe replaces the broken
pieces rather than configuring around them.
"""

from koe.text.normalize import (
    DISPLAY,
    SCORING,
    NormalizationConfig,
    Normalizer,
    normalize_for_display,
    normalize_for_scoring,
)
from koe.text.numbers import contains_numeral, parse_kanji_number, to_arabic
from koe.text.script import (
    Language,
    Script,
    ScriptProfile,
    Span,
    classify,
    code_switch_ratio,
    detect_language,
    primary_language,
    profile,
    spans,
)
from koe.text.tokenize import (
    BilingualTokenizer,
    CharacterTokenizer,
    MeCabTokenizer,
    Token,
    Tokenizer,
    WhitespaceTokenizer,
    japanese_tokenizer,
    mecab_available,
    surfaces,
    tokenizer_for,
)

__all__ = [
    "DISPLAY",
    "SCORING",
    "BilingualTokenizer",
    "CharacterTokenizer",
    "Language",
    "MeCabTokenizer",
    "NormalizationConfig",
    "Normalizer",
    "Script",
    "ScriptProfile",
    "Span",
    "Token",
    "Tokenizer",
    "WhitespaceTokenizer",
    "classify",
    "code_switch_ratio",
    "contains_numeral",
    "detect_language",
    "japanese_tokenizer",
    "mecab_available",
    "normalize_for_display",
    "normalize_for_scoring",
    "parse_kanji_number",
    "primary_language",
    "profile",
    "spans",
    "surfaces",
    "to_arabic",
    "tokenizer_for",
]
