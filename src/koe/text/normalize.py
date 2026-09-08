"""Text normalization for Japanese/English ASR output.

Two different jobs get confused constantly, so koe keeps them apart:

**Normalization for scoring** makes two transcripts comparable. It is
aggressive and lossy -- punctuation goes, case folds, Japanese whitespace
disappears -- because none of that is what an ASR model is being judged on.
Applied identically to reference and hypothesis, it removes formatting from the
error rate. Skip it and a model that writes ``2025年`` scores worse than an
identical model that writes ``二〇二五年``, which tells you nothing about either.

**Normalization for display** makes text correct for a human reader. It is
conservative: fix width and encoding artifacts, leave punctuation and casing
alone. Running the scoring profile on user-visible output would strip the 。and
、 that make Japanese readable.

The Japanese-specific work here:

* **Width folding** (NFKC) -- ``ＫＰＩ`` → ``KPI``, ``ｶﾀｶﾅ`` → ``カタカナ``.
  Vendors disagree on width and the difference is never meaningful.
* **Whitespace removal, but only around Japanese.** Japanese has no word
  spaces, so any an ASR inserts are arbitrary and must go. English spaces are
  load-bearing and must stay. A naive ``.replace(" ", "")`` turns
  ``hello world`` into ``helloworld``; the rule here only deletes a space when
  Japanese script sits on one side of it.
* **Numeral folding**, gated on morphology so ``一般`` does not become ``1般``.
* **Filler removal** -- えーと, あの, um, uh. Disfluencies are a transcription
  policy choice, not a recognition error, so scoring them punishes whichever
  model happens to disagree with the reference's convention.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Literal

from koe.text import numbers
from koe.text.script import Language, Script, classify, primary_language
from koe.text.tokenize import Token, mecab_available, tokenizer_for

# Japanese fillers (えーと / あのー / まあ) and their English counterparts.
_JA_FILLERS = (
    "えーと", "えっと", "えーっと", "ええと", "あのー", "あの", "そのー",
    "まあ", "まぁ", "なんか", "ええ", "うーん", "んー", "はい、えー", "えー",
)  # fmt: skip
_EN_FILLERS = ("um", "uh", "erm", "er", "hmm", "mhm", "like", "you know")

_JA_FILLER_RE = re.compile("|".join(sorted(_JA_FILLERS, key=len, reverse=True)))
_EN_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in sorted(_EN_FILLERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Prolonged sound mark variants that vendors emit interchangeably.
_CHOONPU_VARIANTS = str.maketrans({"─": "ー", "―": "ー", "‐": "ー", "−": "ー", "ｰ": "ー"})

# Typographic quotes folded to ASCII so the intra-word apostrophe rule sees a
# single form regardless of which vendor produced the text.
_QUOTE_VARIANTS = str.maketrans({"’": "'", "‘": "'", "＇": "'"})

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHOONPU_RE = re.compile(r"ー{2,}")


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Which normalization steps to apply."""

    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = "NFKC"
    fold_case: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    strip_japanese_spaces: bool = True
    convert_numerals: bool = True
    normalize_prolonged_sound: bool = True
    strip_fillers: bool = True

    @classmethod
    def for_scoring(cls) -> NormalizationConfig:
        """Aggressive profile: everything not under test is removed."""
        return cls()

    @classmethod
    def for_display(cls) -> NormalizationConfig:
        """Conservative profile: repair encoding artifacts, preserve reading."""
        return cls(
            fold_case=False,
            strip_punctuation=False,
            strip_japanese_spaces=False,
            convert_numerals=False,
            strip_fillers=False,
        )

    def with_(self, **changes: object) -> NormalizationConfig:
        return dataclass_replace(self, **changes)  # type: ignore[arg-type]


def _strip_spaces_around_japanese(text: str) -> str:
    """Delete whitespace adjacent to Japanese script, keep English spacing.

    ``KPI を レビュー`` -> ``KPIをレビュー`` while ``hello world`` is untouched.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            out.append(ch)
            continue
        prev_ch = next((c for c in reversed(text[:i]) if not c.isspace()), "")
        next_ch = next((c for c in text[i + 1 :] if not c.isspace()), "")
        prev_ja = bool(prev_ch) and classify(prev_ch).is_japanese
        next_ja = bool(next_ch) and classify(next_ch).is_japanese
        if prev_ja or next_ja:
            continue  # arbitrary ASR-inserted space
        out.append(" ")
    return "".join(out)


#: Punctuation that carries meaning inside a word and must survive stripping.
#: Removing the apostrophe from ``we're`` yields ``were`` -- a different word,
#: so the "harmless" strip silently invents a substitution error.
_INTRA_WORD_PUNCT = frozenset("'-")


def _strip_punctuation(text: str) -> str:
    out: list[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if classify(ch) is not Script.PUNCT:
            out.append(ch)
            continue
        if (
            ch in _INTRA_WORD_PUNCT
            and 0 < i < last
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            out.append(ch)
    return "".join(out)


def _fold_numeral_run(parts: list[str]) -> str:
    """Parse a run of adjacent numeral tokens as a single number."""
    run = "".join(parts)
    value = numbers.parse_kanji_number(run)
    return str(value) if value is not None else run


def _convert_numerals(text: str, language: Language) -> str:
    """Fold Japanese numerals to Arabic digits, using morphology when present."""
    if not numbers.contains_numeral(text):
        return text

    if not mecab_available():
        return numbers.to_arabic(text)

    tokenizer = tokenizer_for(Language.JA)
    tokens: list[Token] = tokenizer.tokenize(text)
    # Character segmentation carries no POS, so it cannot gate safely.
    if not any(tok.pos for tok in tokens):
        return numbers.to_arabic(text)

    # Adjacent numeral tokens must be parsed as one number. MeCab segments
    # 二千二十五 into 二千 / 二十 / 五, and converting those independently
    # concatenates to "2000205" instead of 2025.
    out: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if tok.is_numeral:
            run.append(tok.surface)
            continue
        if run:
            out.append(_fold_numeral_run(run))
            run = []
        out.append(tok.surface)
    if run:
        out.append(_fold_numeral_run(run))
    return "".join(out)


def _strip_fillers(text: str, language: Language) -> str:
    if language is Language.EN:
        return _EN_FILLER_RE.sub(" ", text)
    text = _JA_FILLER_RE.sub("", text)
    return _EN_FILLER_RE.sub(" ", text)


class Normalizer:
    """Applies a :class:`NormalizationConfig` to text.

    Stateless and cheap; safe to share across sessions.
    """

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig.for_scoring()

    def normalize(self, text: str, language: Language | None = None) -> str:
        """Normalize `text`, detecting its language when not supplied."""
        if not text:
            return ""

        cfg = self.config
        lang = language or primary_language(text)

        if cfg.unicode_form:
            text = unicodedata.normalize(cfg.unicode_form, text)

        text = text.translate(_QUOTE_VARIANTS)

        if cfg.normalize_prolonged_sound:
            text = text.translate(_CHOONPU_VARIANTS)
            text = _REPEATED_CHOONPU_RE.sub("ー", text)

        if cfg.strip_fillers:
            text = _strip_fillers(text, lang)

        if cfg.convert_numerals and lang is not Language.EN:
            text = _convert_numerals(text, lang)

        if cfg.fold_case:
            text = text.casefold()

        if cfg.strip_punctuation:
            text = _strip_punctuation(text)

        if cfg.collapse_whitespace:
            text = _WHITESPACE_RE.sub(" ", text).strip()

        if cfg.strip_japanese_spaces:
            text = _strip_spaces_around_japanese(text)

        return text.strip()

    def __call__(self, text: str, language: Language | None = None) -> str:
        return self.normalize(text, language)


#: Shared instances for the two standard profiles.
SCORING = Normalizer(NormalizationConfig.for_scoring())
DISPLAY = Normalizer(NormalizationConfig.for_display())


def normalize_for_scoring(text: str, language: Language | None = None) -> str:
    """Normalize `text` so two transcripts can be compared fairly."""
    return SCORING.normalize(text, language)


def normalize_for_display(text: str, language: Language | None = None) -> str:
    """Normalize `text` for a human reader, preserving punctuation and case."""
    return DISPLAY.normalize(text, language)
