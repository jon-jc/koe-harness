"""Golden datasets for evaluation.

A case may carry either a path to real audio or a *script* that drives the mock
providers. Supporting both is what lets the eval harness run end-to-end on a
clean checkout with no data and no credentials -- so the machinery is exercised
by CI on every commit, and swapping in real recordings later changes the data,
not the code.

Cases are **tagged**, and tags are the point. A single corpus-level CER is
nearly useless for improving a product: it averages away the fact that
code-switched utterances are several times worse than monolingual ones, which
is precisely the actionable finding. The runner reports every slice separately.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from koe.domain.transcript import SpeakerTurn
from koe.providers.mock import ScriptedUtterance
from koe.text.script import Language, primary_language


class ScriptLine(BaseModel):
    """One scripted utterance, in the serializable form."""

    speaker: str
    text: str
    start: float
    end: float

    def to_utterance(self) -> ScriptedUtterance:
        return ScriptedUtterance(
            speaker=self.speaker, text=self.text, start=self.start, end=self.end
        )


class EvalCase(BaseModel):
    """One scored item: audio (or a script), a reference, and its labels."""

    id: str
    reference: str = ""
    language: Language = Language.UNKNOWN
    audio_path: str | None = None
    script: list[ScriptLine] = Field(default_factory=list)
    speakers: list[SpeakerTurn] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    def _raw_script_text(self) -> str:
        """Script lines concatenated with no joiner, for language detection only.

        Language detection has to happen *before* the joiner is chosen (the
        joiner depends on the language), so it runs against this raw form. Doing
        it the obvious way -- asking :meth:`resolved_language`, which asks
        :meth:`resolved_reference`, which asks for the joiner -- is a cycle.
        """
        return "".join(line.text for line in self.script)

    def resolved_language(self) -> Language:
        if self.language is not Language.UNKNOWN:
            return self.language
        return primary_language(self.reference or self._raw_script_text())

    def utterances(self) -> list[ScriptedUtterance]:
        return [line.to_utterance() for line in self.script]

    def resolved_reference(self) -> str:
        """Reference text, derived from the script when not given explicitly.

        Uses the same joiner rule as :attr:`Transcript.text` so a reference and
        a hypothesis over identical content are byte-identical rather than
        differing by whitespace that the scorer would then have to absorb.
        """
        if self.reference:
            return self.reference
        if not self.script:
            return ""
        joiner = "" if primary_language(self._raw_script_text()) is Language.JA else " "
        return joiner.join(line.text for line in self.script)

    def duration(self) -> float:
        if self.script:
            return max(line.end for line in self.script)
        if self.speakers:
            return max(turn.end for turn in self.speakers)
        return 0.0


class Dataset(BaseModel):
    """A named collection of evaluation cases."""

    name: str = "unnamed"
    description: str = ""
    cases: list[EvalCase] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:  # type: ignore[override]
        return iter(self.cases)

    @property
    def tags(self) -> list[str]:
        return sorted({tag for case in self.cases for tag in case.tags})

    @property
    def languages(self) -> list[Language]:
        return sorted({case.resolved_language() for case in self.cases})

    def filter(
        self,
        *,
        language: Language | None = None,
        tags: Sequence[str] | None = None,
    ) -> Dataset:
        """Narrow to a slice, for scoring one dimension at a time."""
        selected = [
            case
            for case in self.cases
            if (language is None or case.resolved_language() is language)
            and (tags is None or set(tags) <= set(case.tags))
        ]
        return Dataset(name=self.name, description=self.description, cases=selected)

    # -- persistence ---------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        """Load a JSONL dataset -- one JSON object per line.

        JSONL rather than a single JSON array so a corpus can be appended to and
        streamed, and so a malformed line is a single bad case rather than an
        unparseable file.
        """
        file = Path(path)
        cases: list[EvalCase] = []
        with file.open(encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                try:
                    cases.append(EvalCase.model_validate_json(stripped))
                except Exception as exc:
                    raise ValueError(f"{file}:{lineno}: invalid eval case: {exc}") from exc
        return cls(name=name or file.stem, cases=cases)

    def to_jsonl(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(case.model_dump_json(exclude_defaults=True) + "\n")

    @classmethod
    def from_scripts(
        cls,
        name: str,
        scripts: dict[str, Sequence[ScriptedUtterance]],
        *,
        tags: dict[str, list[str]] | None = None,
    ) -> Dataset:
        """Build a dataset from in-memory scripts, for tests and smoke runs."""
        tag_map = tags or {}
        cases = []
        for case_id, utterances in scripts.items():
            lines = [
                ScriptLine(speaker=u.speaker, text=u.text, start=u.start, end=u.end)
                for u in utterances
            ]
            cases.append(
                EvalCase(
                    id=case_id,
                    script=lines,
                    speakers=[
                        SpeakerTurn(speaker=u.speaker, start=u.start, end=u.end) for u in utterances
                    ],
                    tags=tag_map.get(case_id, []),
                )
            )
        return cls(name=name, cases=cases)


def write_jsonl(path: str | Path, records: Sequence[dict[str, object]]) -> None:
    """Write arbitrary records as JSONL, used for run artifacts."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
