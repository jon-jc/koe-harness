"""The user vocabulary, as a plugin.

Wiring only: :mod:`koe.text.vocabulary` holds the rules and knows nothing about
the kernel, and this module puts them where the pipeline can find them.

**A plugin rather than a field on the session**, for the reason the rest of the
harness is: the pipeline asks ``ctx.get("vocabulary")`` and gets ``None`` when
this is not mounted, which is the pre-existing behaviour rather than an error.
Turning the plugin off gives back a recognizer that has never heard of the
user's colleagues -- degraded, but working -- instead of a stack trace.

**The service is the store, not the compiled vocabulary.** A caller holds a
reference for the life of a session, and edits to the word list have to take
effect on the next utterance without anything re-registering. Handing out the
compiled object would freeze whichever version happened to be current when the
session opened.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from koe.text.vocabulary import EMPTY, Vocabulary

logger = logging.getLogger(__name__)

#: The word list, in the import format, in the user's config directory. A plain
#: text file on purpose: it is a list of words, people already know how to edit
#: one of those, and it can be kept in a dotfiles repository.
FILENAME = "vocabulary.txt"


class VocabularyStore:
    """The user's word list, on disk and compiled in memory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._vocabulary = EMPTY
        self.reload()

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    @property
    def terms(self) -> tuple[str, ...]:
        return self._vocabulary.terms

    def __len__(self) -> int:
        return len(self._vocabulary)

    def apply(self, text: str) -> str:
        """Correct one utterance. The hot path: called per final segment."""
        return self._vocabulary.apply(text)

    def prompt_hint(self) -> str:
        return self._vocabulary.prompt_hint()

    def text(self) -> str:
        return self._vocabulary.to_text()

    def reload(self) -> Vocabulary:
        """Re-read the file. A missing one is an empty vocabulary, not an error."""
        if self.path is None or not self.path.is_file():
            self._vocabulary = EMPTY
            return self._vocabulary
        try:
            source = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            # A word list that cannot be read must not take the session with
            # it: the cost of ignoring it is some misrecognized proper nouns.
            logger.warning("could not read %s: %s", self.path, exc)
            self._vocabulary = EMPTY
            return self._vocabulary
        self._vocabulary = Vocabulary.parse(source)
        return self._vocabulary

    def save(self, source: str) -> Vocabulary:
        """Replace the word list and recompile it."""
        parsed = Vocabulary.parse(source)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written back in the canonical format rather than as typed, so the
            # file and the editor agree about what was saved.
            self.path.write_text(parsed.to_text() + "\n", encoding="utf-8")
        self._vocabulary = parsed
        return parsed


def user_vocabulary(ctx: Any, config: Any = None) -> None:
    """Provide the `vocabulary` service."""
    path = getattr(config, "path", None) if config is not None else None
    if path is None:
        from koe.desktop.paths import config_dir

        path = config_dir() / FILENAME

    store = VocabularyStore(Path(path))
    ctx.provide("vocabulary", store, replace=True)
    logger.info("user vocabulary mounted from %s (%d entries)", store.path, len(store))
