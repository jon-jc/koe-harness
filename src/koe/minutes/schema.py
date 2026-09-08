"""The 議事録 (meeting minutes) schema.

The shape of this schema is the main quality mechanism in the LLM layer, not
just a serialization format.

**Every extracted claim carries a verbatim `source_quote`.** A decision, an
action item, a due date -- each one must be accompanied by the span of
transcript it came from. That single requirement converts hallucination
detection from "ask another model whether this looks right" into a string
operation: either the quote is in the transcript or it is not.

That matters because the failure mode here is specific and expensive. An LLM
asked to summarize a meeting will, occasionally and very fluently, invent an
action item that nobody agreed to, or attach a deadline nobody said. In a
minutes product that lands in someone's task list. Requiring a citation makes
the invention detectable *without* a second model call, without an LLM judge's
own uncertainty, and at effectively zero cost.

It also improves the output on its own: a model that has to cite is measurably
less inclined to embellish, because there is nowhere to put the embellishment.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from koe.text.script import Language


class SourcedClaim(BaseModel):
    """Base for anything the model asserts about the meeting."""

    source_quote: str = Field(
        default="",
        description=(
            "The exact span of transcript this came from, copied verbatim. "
            "Do not paraphrase, translate, or clean it up."
        ),
    )
    speaker: str = Field(default="", description="Who said it, as labelled in the transcript.")


class Decision(SourcedClaim):
    """Something the meeting concluded."""

    statement: str = Field(description="The decision, stated in one sentence.")


class ActionItem(SourcedClaim):
    """A commitment made by a named participant."""

    task: str = Field(description="What is to be done, in one sentence.")
    owner: str = Field(
        default="",
        description=(
            "Who committed to it, exactly as they are labelled in the transcript. "
            "Leave empty if nobody explicitly took it on -- do not guess."
        ),
    )
    due: str = Field(
        default="",
        description=(
            "The deadline exactly as stated (e.g. 金曜日, 3月10日, next week). "
            "Leave empty if none was given -- do not infer one."
        ),
    )


class Topic(BaseModel):
    """One discussion thread."""

    title: str = Field(description="Short label for the topic.")
    summary: str = Field(description="What was discussed, in two or three sentences.")


class Minutes(BaseModel):
    """A complete 議事録."""

    language: Language = Language.UNKNOWN
    title: str = Field(default="", description="A short title for the meeting.")
    participants: list[str] = Field(
        default_factory=list, description="Speakers, as labelled in the transcript."
    )
    summary: str = Field(default="", description="Three or four sentences covering the meeting.")
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)

    def claims(self) -> list[SourcedClaim]:
        """Every claim that must be supported by a quote."""
        return [*self.decisions, *self.action_items]

    def render(self) -> str:
        """Human-readable minutes, in the meeting's own language."""
        ja = self.language in (Language.JA, Language.MIXED)
        labels = (
            {
                "participants": "出席者",
                "summary": "概要",
                "topics": "議題",
                "decisions": "決定事項",
                "actions": "アクションアイテム",
                "owner": "担当",
                "due": "期限",
                "unassigned": "未割当",
            }
            if ja
            else {
                "participants": "Participants",
                "summary": "Summary",
                "topics": "Topics",
                "decisions": "Decisions",
                "actions": "Action items",
                "owner": "Owner",
                "due": "Due",
                "unassigned": "unassigned",
            }
        )

        lines: list[str] = []
        if self.title:
            lines += [f"# {self.title}", ""]
        if self.participants:
            lines += [
                f"**{labels['participants']}**: {'、'.join(self.participants) if ja else ', '.join(self.participants)}",
                "",
            ]
        if self.summary:
            lines += [f"## {labels['summary']}", "", self.summary, ""]
        if self.topics:
            lines += [f"## {labels['topics']}", ""]
            for topic in self.topics:
                lines += [f"### {topic.title}", "", topic.summary, ""]
        if self.decisions:
            lines += [f"## {labels['decisions']}", ""]
            lines += [f"- {d.statement}" for d in self.decisions]
            lines.append("")
        if self.action_items:
            lines += [f"## {labels['actions']}", ""]
            for item in self.action_items:
                owner = item.owner or labels["unassigned"]
                due = f" / {labels['due']}: {item.due}" if item.due else ""
                lines.append(f"- [ ] {item.task} — {labels['owner']}: {owner}{due}")
            lines.append("")
        return "\n".join(lines).strip()


def minutes_json_schema() -> dict[str, Any]:
    """JSON Schema for constrained decoding.

    ``additionalProperties: false`` is forced throughout: without it a model can
    return extra keys that validate but silently carry content nothing reads,
    which is a quiet way to lose information the user was shown.
    """
    schema = Minutes.model_json_schema()
    _harden(schema)
    return schema


def _harden(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _harden(value)
    elif isinstance(node, list):
        for item in node:
            _harden(item)
