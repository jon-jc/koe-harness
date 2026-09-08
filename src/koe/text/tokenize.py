"""Tokenization for Japanese and English.

English tokenizes on whitespace. Japanese does not have whitespace, so "words"
only exist relative to a segmenter -- which means **a Japanese WER is only
meaningful alongside the name of the tokenizer that produced it**. Two teams
reporting 12% WER on the same audio with different segmenters have not measured
the same thing.

koe therefore treats the tokenizer as part of the metric's identity: every
tokenizer reports an :attr:`~Tokenizer.name`, and the evaluation layer records
it in the result. When MeCab is unavailable we fall back to character
segmentation and say so, rather than quietly producing a number that looks
comparable and isn't.

MeCab also gives part-of-speech tags, which the normalizer uses to decide
whether 一 in a given position is the number one or the first character of
一般. That is not recoverable from characters alone.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from koe.text.script import Language, Script, classify, primary_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Token:
    """A single token, with morphology when the backend provides it."""

    surface: str
    pos: str = ""
    lemma: str = ""
    reading: str = ""

    @property
    def is_numeral(self) -> bool:
        """Whether morphology tagged this token as a number.

        Both UniDic (``名詞,数詞``) and IPADIC (``名詞,数``) are recognised.
        """
        return "数" in self.pos

    @property
    def is_punctuation(self) -> bool:
        return any(tag in self.pos for tag in ("補助記号", "記号")) or all(
            classify(ch) is Script.PUNCT for ch in self.surface
        )


@runtime_checkable
class Tokenizer(Protocol):
    """Segments text into tokens."""

    name: str

    def tokenize(self, text: str) -> list[Token]: ...


class WhitespaceTokenizer:
    """English tokenizer: split on whitespace, strip edge punctuation."""

    name = "whitespace"

    def tokenize(self, text: str) -> list[Token]:
        return [Token(surface=w) for w in text.split() if w]


class CharacterTokenizer:
    """Fallback Japanese tokenizer: one token per character.

    Character segmentation makes WER collapse into CER. That is a defensible
    metric for Japanese -- it is what most published Japanese ASR results
    report -- but it is *not* the same number as a MeCab-segmented WER, which
    is exactly why this reports a distinct name.
    """

    name = "character"

    def tokenize(self, text: str) -> list[Token]:
        return [Token(surface=ch) for ch in text if not ch.isspace()]


@dataclass
class MeCabTokenizer:
    """Morphological analysis via fugashi/MeCab.

    Requires the ``ja`` extra (``pip install 'koe-harness[ja]'``).
    """

    name: str = field(default="mecab-unidic", init=False)
    _tagger: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # optional dependency, imported on demand so the base install stays light
        import fugashi

        self._tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> list[Token]:
        if not text:
            return []
        tokens: list[Token] = []
        for word in self._tagger(text):  # type: ignore[operator]
            feature = word.feature
            pos = ",".join(
                str(part)
                for part in (
                    getattr(feature, "pos1", "") or "",
                    getattr(feature, "pos2", "") or "",
                )
                if part and part != "*"
            )
            tokens.append(
                Token(
                    surface=word.surface,
                    pos=pos,
                    lemma=str(getattr(feature, "lemma", "") or ""),
                    reading=str(getattr(feature, "kana", "") or ""),
                )
            )
        return tokens


@functools.lru_cache(maxsize=1)
def mecab_available() -> bool:
    """Whether MeCab-backed tokenization can be used in this environment."""
    try:
        import fugashi  # noqa: F401

        MeCabTokenizer()
    except Exception as exc:  # noqa: BLE001 - any import/dictionary failure disqualifies it
        logger.info("MeCab unavailable (%s); Japanese will use character segmentation", exc)
        return False
    return True


@functools.lru_cache(maxsize=4)
def japanese_tokenizer(*, prefer_mecab: bool = True) -> Tokenizer:
    """Best available Japanese tokenizer."""
    if prefer_mecab and mecab_available():
        return MeCabTokenizer()
    return CharacterTokenizer()


class BilingualTokenizer:
    """Dispatches per text to the tokenizer appropriate for its language.

    Mixed strings are segmented as Japanese: MeCab handles embedded Latin runs
    as single tokens, which is the behaviour we want for ``KPIをレビュー``.
    """

    def __init__(self, *, prefer_mecab: bool = True) -> None:
        self._ja = japanese_tokenizer(prefer_mecab=prefer_mecab)
        self._en = WhitespaceTokenizer()
        self.name = f"bilingual({self._ja.name}+{self._en.name})"

    def tokenize(self, text: str, language: Language | None = None) -> list[Token]:
        lang = language or primary_language(text)
        if lang is Language.EN:
            return self._en.tokenize(text)
        return self._ja.tokenize(text)


def tokenizer_for(language: Language, *, prefer_mecab: bool = True) -> Tokenizer:
    """Return the tokenizer koe uses for `language`."""
    if language is Language.EN:
        return WhitespaceTokenizer()
    return japanese_tokenizer(prefer_mecab=prefer_mecab)


def surfaces(tokens: Sequence[Token]) -> list[str]:
    """Extract surface forms, the form the metrics compare."""
    return [t.surface for t in tokens]
