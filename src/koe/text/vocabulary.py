"""A user vocabulary: the words a general ASR model has no way to know.

Every recognizer is wrong about the same class of word -- the ones specific to
the people using it. Colleagues' names, the product, the internal acronym, the
client. No amount of provider-switching fixes this, because the information is
not in any model; it is in the user's head. OpenWhispr's answer is a
user-editable dictionary, and it is the right one.

Two things live here, because the same list serves both and keeping them apart
would mean maintaining the list twice:

**Terms** are hints, sent to the recognizer *before* it decodes. Most ASR APIs
accept a biasing list or a prompt, and a term supplied up front is recognized
correctly rather than corrected afterwards. This is strictly better when it
works, and it is not available on every provider.

**Corrections** are ``heard => written`` rules applied *after* decoding. This
is the fallback, and it is the only option when the recognizer has already
committed to a plausible wrong answer.

**Matching differs by script, and this is the part that a Latin-first
implementation gets wrong.** A Latin term is matched at word boundaries, so a
rule for "koe" does not fire inside "invoke". Japanese has no word boundaries
to match on -- ``\\b`` is defined in terms of ``\\w``, and between two kanji
there is no boundary for it to find, so a boundary-anchored Japanese rule
matches nothing at all. Japanese terms are therefore matched as plain
substrings, which is what the script actually supports.

**Every rule is applied in a single pass.** Replacing one at a time lets the
output of one rule feed the input of the next: with ``A => B`` and ``B => C``,
a sequential pass turns A into C, which nobody asked for and which depends on
the order the rules happen to be stored in. One alternation over all patterns,
longest first, cannot cascade.

The import format is adapted from OpenWhispr's ``parseDictionaryImportText``
(MIT), extended so that one list can carry both kinds of entry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from koe.text.script import classify

#: Accepted between the heard form and the written one. ``=>`` is the documented
#: spelling; the others are what people type anyway.
_ARROWS = ("=>", "->", "→", "⇒")


def _has_japanese(text: str) -> bool:
    return any(classify(char).is_japanese for char in text)


@dataclass(frozen=True, slots=True)
class Entry:
    """One vocabulary item.

    `written` empty means the entry is a hint only: tell the recognizer this
    word exists, but do not rewrite anything if it comes back different.
    """

    heard: str
    written: str = ""

    @property
    def corrects(self) -> bool:
        return bool(self.written) and self.written != self.heard

    @property
    def term(self) -> str:
        """The spelling to bias the recognizer toward.

        The written form when there is one: biasing toward the misrecognition
        would be asking for the very mistake the entry exists to fix.
        """
        return self.written or self.heard


@dataclass(slots=True)
class Vocabulary:
    """A user's terms and corrections, compiled once and applied per utterance."""

    entries: tuple[Entry, ...] = ()
    _pattern: re.Pattern[str] | None = field(default=None, init=False, repr=False)
    _replacements: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compile()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    @property
    def terms(self) -> tuple[str, ...]:
        """Every spelling worth biasing the recognizer toward."""
        seen: dict[str, None] = {}
        for entry in self.entries:
            if entry.term:
                seen.setdefault(entry.term, None)
        return tuple(seen)

    def _compile(self) -> None:
        """Build the single alternation that applies every correction at once."""
        corrections = [entry for entry in self.entries if entry.corrects]
        if not corrections:
            self._pattern = None
            self._replacements = {}
            return

        # Longest first, so "KPI ダッシュボード" wins over a bare "KPI" that
        # would otherwise match its prefix and strand the rest.
        corrections.sort(key=lambda entry: len(entry.heard), reverse=True)

        branches: list[str] = []
        for entry in corrections:
            escaped = re.escape(entry.heard)
            if _has_japanese(entry.heard):
                # No word boundaries exist between kanji, so anchoring on them
                # would match nothing. A substring is what the script offers.
                branches.append(escaped)
            else:
                branches.append(rf"\b{escaped}\b")
            self._replacements[entry.heard.casefold()] = entry.written

        self._pattern = re.compile("|".join(branches), re.IGNORECASE)

    def apply(self, text: str) -> str:
        """Rewrite `text` according to the corrections, in one pass."""
        if not text or self._pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            return self._replacements.get(match.group(0).casefold(), match.group(0))

        return self._pattern.sub(replace, text)

    def prompt_hint(self, *, limit: int = 200) -> str:
        """The terms as a biasing prompt, for providers that take one.

        Capped because these go into a prompt field with a length limit on most
        providers, and one silently truncated by the vendor drops whichever
        terms happened to sort last.
        """
        return ", ".join(self.terms[:limit])

    @classmethod
    def parse(cls, text: str) -> Vocabulary:
        """Read the import format: one entry per line or comma-separated.

        A line with an arrow is a correction, ``heard => written``. A line
        without one is a term: recognize this word, do not rewrite anything.
        Blank entries and ``#`` comments are dropped.
        """
        entries: list[Entry] = []
        seen: set[str] = set()

        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # A line containing an arrow is one rule, even if the written form
            # contains a comma; only arrow-free lines are split on commas.
            chunks = [line] if any(arrow in line for arrow in _ARROWS) else line.split(",")
            for chunk in chunks:
                entry = _parse_entry(chunk)
                if entry is None or entry.heard.casefold() in seen:
                    continue
                seen.add(entry.heard.casefold())
                entries.append(entry)

        return cls(entries=tuple(entries))

    def to_text(self) -> str:
        """Render back to the import format, so a round trip is lossless."""
        lines = []
        for entry in self.entries:
            lines.append(f"{entry.heard} => {entry.written}" if entry.corrects else entry.heard)
        return "\n".join(lines)

    @classmethod
    def of(cls, items: Iterable[Entry | tuple[str, str] | str]) -> Vocabulary:
        """Build one from entries, pairs, or bare terms."""
        entries: list[Entry] = []
        for item in items:
            if isinstance(item, Entry):
                entries.append(item)
            elif isinstance(item, tuple):
                entries.append(Entry(heard=item[0], written=item[1]))
            else:
                entries.append(Entry(heard=item))
        return cls(entries=tuple(entries))


def _parse_entry(chunk: str) -> Entry | None:
    text = chunk.strip()
    if not text:
        return None

    for arrow in _ARROWS:
        if arrow in text:
            heard, _, written = text.partition(arrow)
            heard, written = heard.strip(), written.strip()
            return Entry(heard=heard, written=written) if heard else None

    return Entry(heard=text)


#: An empty vocabulary, so callers can avoid a None check.
EMPTY = Vocabulary()
