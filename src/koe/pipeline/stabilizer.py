"""Partial hypothesis stabilization.

A streaming ASR revises itself. Feed it a growing audio buffer and successive
hypotheses read:

    こんにちは
    こんにちは今日
    こんにちは今日の議
    こんにちは本日の議題は

Rendering each one directly produces text that rewrites itself several times a
second. Beyond looking broken, it is genuinely hard to read -- the eye keeps
re-reading a line that keeps changing, so a live caption that flickers is worse
than one that lags.

The fix is **LocalAgreement** (Macháček et al., used in whisper_streaming):
commit only the prefix that the last *n* hypotheses agree on. Agreement across
independent decodes is good evidence the model has settled, so committed text
almost never needs revising -- and this layer guarantees it never does, because
once text is shown as stable, taking it back is a worse experience than having
waited.

Agreement is computed on **tokens, not characters**, using the same tokenizer
the metrics use. For Japanese that means MeCab units where available: comparing
raw characters would commit half of a word whose second half the model is still
deciding, and 食べ / 食べない differ only in the part that carries the negation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from koe.text.script import Language, primary_language
from koe.text.spacing import join_tokens
from koe.text.tokenize import CharacterTokenizer, surfaces, tokenizer_for


def _word_level(language: Language) -> bool:
    """Whether this language's tokenizer segments on word boundaries.

    Cheap: `tokenizer_for` is memoized, so this is a dict lookup and an
    attribute read rather than a tagger construction.
    """
    return tokenizer_for(language).name != CharacterTokenizer.name


def _join(tokens: list[str], language: Language) -> str:
    """Rejoin tokens, spacing each seam by the scripts that meet at it.

    One rule per *seam*, not one per language. A Japanese utterance containing
    an English phrase -- which is most of them in a Japanese office -- needs the
    space inside "KPI dashboard" and no space around の, and a per-language rule
    can deliver only one of those. Joining a Japanese utterance with "" is what
    turned the vocabulary correction "KPI dashboard" into "KPIdashboard" the
    first time one ran.

    The reconstruction is only attempted when the tokenizer segmented on words.
    Without MeCab the Japanese fallback is one token per character, which
    destroys the evidence rather than merely omitting it -- see
    :func:`koe.text.spacing.join_tokens`.
    """
    return join_tokens(tokens, language, word_level=_word_level(language))


@dataclass(frozen=True, slots=True)
class Stabilized:
    """The result of folding in one new hypothesis."""

    committed: str
    pending: str
    newly_committed: str
    language: Language

    @property
    def full(self) -> str:
        """Everything currently known, stable part first."""
        if not self.pending:
            return self.committed
        if not self.committed:
            return self.pending
        joiner = " " if self.language is Language.EN else ""
        return f"{self.committed}{joiner}{self.pending}"


def common_prefix(sequences: list[list[str]]) -> list[str]:
    """Longest prefix shared by every sequence."""
    if not sequences:
        return []
    shortest = min(len(s) for s in sequences)
    prefix: list[str] = []
    for index in range(shortest):
        token = sequences[0][index]
        if all(seq[index] == token for seq in sequences[1:]):
            prefix.append(token)
        else:
            break
    return prefix


@dataclass(slots=True)
class Stabilizer:
    """Commits the prefix that successive hypotheses agree on.

    Stateful per utterance; call :meth:`reset` at an endpoint.
    """

    #: How many consecutive hypotheses must agree. 2 is the usual choice: 1
    #: commits immediately and flickers, 3+ adds a hypothesis of latency for
    #: little additional stability.
    agreement: int = 2
    language: Language = Language.UNKNOWN

    _history: list[list[str]] = field(default_factory=list, init=False, repr=False)
    _committed: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def committed_text(self) -> str:
        return _join(self._committed, self._language())

    def _language(self) -> Language:
        return self.language if self.language is not Language.UNKNOWN else Language.JA

    def update(self, hypothesis: str) -> Stabilized:
        """Fold in a new partial hypothesis."""
        language = (
            self.language if self.language is not Language.UNKNOWN else primary_language(hypothesis)
        )
        tokens = surfaces(tokenizer_for(language).tokenize(hypothesis))

        self._history.append(tokens)
        if len(self._history) > self.agreement:
            self._history.pop(0)

        newly: list[str] = []
        if len(self._history) >= self.agreement:
            agreed = common_prefix(self._history)
            # Committed text is never revised. The agreed prefix is accepted
            # only when it *extends* what is already committed -- comparing
            # lengths alone is not enough, because two hypotheses can agree on
            # a longer prefix that contradicts the committed one, and taking
            # that would rewrite text the viewer has already read.
            extends = len(agreed) > len(self._committed) and (
                agreed[: len(self._committed)] == self._committed
            )
            if extends:
                newly = agreed[len(self._committed) :]
                self._committed = agreed

        if tokens[: len(self._committed)] == self._committed:
            pending = tokens[len(self._committed) :]
        else:
            # The model has contradicted committed text. Committed stays put;
            # its current guess is shown in full as pending, so the viewer sees
            # the proposed correction marked as unsettled rather than silently
            # applied.
            pending = tokens
        return Stabilized(
            committed=_join(self._committed, language),
            pending=_join(pending, language),
            newly_committed=_join(newly, language),
            language=language,
        )

    def finalize(self, hypothesis: str) -> Stabilized:
        """Commit everything, for a hypothesis known to be final.

        At an endpoint the ASR has seen the whole utterance, so waiting for
        agreement would only add latency to text that will not change.
        """
        language = (
            self.language if self.language is not Language.UNKNOWN else primary_language(hypothesis)
        )
        tokens = surfaces(tokenizer_for(language).tokenize(hypothesis))
        newly = tokens[len(self._committed) :]
        self._committed = tokens
        return Stabilized(
            committed=_join(tokens, language),
            pending="",
            newly_committed=_join(newly, language),
            language=language,
        )

    def reset(self) -> None:
        """Clear state at an utterance boundary."""
        self._history.clear()
        self._committed.clear()
