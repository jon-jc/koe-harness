"""Joining dictated fragments, in a script that uses spaces and one that does not.

Dictation arrives in pieces. Each utterance is recognized on its own and then
appended to whatever is already in the buffer, and the seam between two pieces
is a decision: English wants a space there and Japanese does not. Getting it
wrong is not cosmetic. ``京都 に 行きます`` is not a typo to a Japanese reader,
it is broken text, and the gaps accumulate one per utterance for as long as the
session runs.

Adapted from OpenWhispr's ``src/helpers/smartSpacing.js`` (MIT). The character
ranges are theirs and are well chosen -- in particular **Hangul is deliberately
excluded**, because Korean does separate words with spaces and lumping it in
with Han and kana would break it.

**Where this differs, and why.** OpenWhispr decides from the left side alone:
it appends a trailing space to the text it is about to paste unless that text
ends in CJK. It has to, because at paste time the next dictation does not exist
yet. koe joins two fragments it already holds, so it can look at both sides of
the seam -- which is what the mixed case needs. Japanese business speech
code-switches constantly::

    "Q3"  +  "の売上"

Left-side-only sees Latin on the left, adds a space, and produces ``Q3 の売上``.
Looking at both sides sees kana on the right and joins them. The one-sided rule
is still here as :func:`with_trailing_space`, for the case it was written for:
pasting at a cursor in another application, where there is no right-hand side
to consult.
"""

from __future__ import annotations

from collections.abc import Iterable

from koe.text.script import Language, Script, classify

#: CJK punctuation and the fullwidth/halfwidth forms, which behave like the
#: scripts they sit among rather than like their ASCII lookalikes. A trailing
#: space after ``です。`` is as wrong as one after ``です``.
_UNSPACED_RANGES = (
    (0x3000, 0x303F),  # CJK symbols and punctuation: 、。「」〜・
    (0xFF00, 0xFF65),  # fullwidth forms, halfwidth katakana punctuation
    (0xFE10, 0xFE1F),  # vertical forms
    (0xFE30, 0xFE4F),  # CJK compatibility forms
)

#: ASCII punctuation that closes rather than opens, so a space before it is
#: always wrong. An utterance genuinely can begin with one of these -- a
#: speaker who pauses before ", and then we ship" gets two utterances.
_NO_SPACE_BEFORE = frozenset(".,!?;:)]}%…")

#: The opening halves, where a space after is wrong for the same reason.
_NO_SPACE_AFTER = frozenset("([{#@¥$")


def is_unspaced(char: str) -> bool:
    """Whether `char` belongs to a script that does not separate words.

    Note that fullwidth Latin (ＡＢＣ) counts as unspaced here even though
    :func:`koe.text.script.classify` calls it Latin. The two answers are for
    different questions: for deciding a tokenizer it is Latin, but it only ever
    appears inside Japanese text, and spacing follows the text it appears in.
    """
    if not char:
        return False
    if classify(char).is_japanese:
        return True
    cp = ord(char)
    return any(low <= cp <= high for low, high in _UNSPACED_RANGES)


def needs_space(left: str, right: str) -> bool:
    """Whether a space belongs at the seam between two fragments."""
    if not left or not right:
        return False

    end, start = left[-1], right[0]
    if end.isspace() or start.isspace():
        return False
    if is_unspaced(end) or is_unspaced(start):
        return False
    return start not in _NO_SPACE_BEFORE and end not in _NO_SPACE_AFTER


def join(left: str, right: str) -> str:
    """Append `right` to `left`, spacing the seam the way the scripts want."""
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}" if needs_space(left, right) else f"{left}{right}"


def join_all(fragments: Iterable[str]) -> str:
    """Fold :func:`join` over a sequence of fragments."""
    out = ""
    for fragment in fragments:
        out = join(out, fragment)
    return out


def join_tokens(tokens: Iterable[str], language: Language, *, word_level: bool = True) -> str:
    """Rejoin *tokenizer output*, which is a different problem from :func:`join`.

    :func:`join` sees two fragments a person actually said, and the space
    between them is a fact about the text. Tokens are not that: the tokenizer
    has already discarded the original spacing, so this is reconstruction, and
    reconstruction is only possible to the extent the segmentation preserved
    the evidence.

    English keeps ``" ".join``, because that is exactly what its tokenizer's
    output is: words that had spaces between them.

    Japanese joins tightly **except where two Latin words meet**. MeCab emits
    ``KPI`` and ``dashboard`` as separate tokens with no record that a space
    stood between them, and joining tightly gives ``KPIdashboard``. Anchoring
    that exception on *letters* rather than on "not Japanese" is what keeps
    ``Q`` + ``3`` -- which MeCab also splits, out of a string that never had a
    space in it -- from becoming ``Q 3``. A digit is not a word.

    ``word_level=False`` turns the exception off, and callers using a
    character-granularity tokenizer must pass it. Under
    :class:`~koe.text.tokenize.CharacterTokenizer` every token is one
    character, so "two Latin tokens meet" is true between every pair of letters
    in a word and the rule would render ``KPI dashboard`` as
    ``K P I d a s h b o a r d``. There is nothing to reconstruct from in that
    case: the segmentation threw the spaces away and left no boundary that
    means anything. Joining tightly is the honest answer, and it is what koe
    did everywhere before this function existed.
    """
    parts = [part for part in tokens if part]
    if not parts:
        return ""
    if language is Language.EN:
        return " ".join(parts)
    if not word_level:
        return "".join(parts)

    out = parts[0]
    for part in parts[1:]:
        joiner = " " if _is_latin_letter(out[-1]) and _is_latin_letter(part[0]) else ""
        out = f"{out}{joiner}{part}"
    return out


def _is_latin_letter(char: str) -> bool:
    return classify(char) is Script.LATIN


def with_trailing_space(text: str) -> str:
    """OpenWhispr's original one-sided rule, for pasting at a cursor.

    Use this only where the following text is genuinely unknowable -- pasting
    into another application. Within koe's own buffer, :func:`join` is strictly
    better informed, because there the right-hand side is in hand.
    """
    if not text:
        return text
    if text[-1].isspace() or is_unspaced(text[-1]):
        return text
    return text + " "
