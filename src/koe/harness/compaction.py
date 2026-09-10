"""Compaction: making room without losing the record.

Ported from DeepSeek Harness's `dsh-compaction`, `compaction-basic` and
`compaction-tool-result-pruner` (MIT). koe had none of this, so a long meeting
chat simply ran into the context window and the provider refused the request.

Four positions, all of them dsh's, and each one is a thing a naive
implementation gets wrong.

**Compaction shadows, it does not delete.** The replacement is appended and
the range it stands for is marked shadowed. Both remain in the log, so
"what was compacted away" has an answer, the operation replays, and a bug in
the summarizer costs a bad summary rather than a lost conversation.

**A replacement lands at the position of the range it replaces**, not at the
end of the log. A summary written now, for a conversation that happened an hour
ago, belongs where that conversation was. Appending it at the end would reorder
the conversation and put the summary of the beginning after the middle.

**A cut is only legal where no unanswered tool call crosses it.** This is the
rule that makes compaction safe, and the reason it is computed from content
rather than from step markers: compaction moves surface positions, so a cut
derived from "where step 4 started" is meaningless after the first replacement.
An assistant message opens as many calls as it contains and each `tool/result`
closes one; a cut is balanced exactly where the count is zero. Cutting anywhere
else produces an assistant turn whose calls are answered by results the request
no longer contains, which every vendor rejects.

**The bracket is a lock, and an unmatched start is meant to be visible.**
`compaction/start` … `compaction/end` encloses the whole transaction, including
the summarizer's model call. A crash between them leaves an unmatched start in
the log rather than a silently half-compacted session, so the damage is
detectable instead of merely done.

Two backends, as in dsh:

* **Summarizing** — asks the model to write a summary of the range. Costs a
  request; keeps the meaning.
* **Pruning** — replaces the middle of oversized tool results with a marker, no
  model involved. Free and deterministic. A 40,000-character file read is
  almost never needed in full three turns later, and pruning it first often
  removes the pressure without spending anything.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from koe.harness.session import (
    ASSISTANT_MESSAGE,
    COMPACTION_END,
    COMPACTION_PRUNE,
    COMPACTION_START,
    COMPACTION_SUMMARY,
    SYSTEM_MESSAGE,
    TOOL_RESULT,
    USER_MESSAGE,
    Session,
)
from koe.harness.tokens import Measurement, TokenMeter, estimate_text

logger = logging.getLogger(__name__)

#: Compact when the surface reaches this fraction of the window. dsh's default.
#: Below 1.0 by a wide margin on purpose: compaction itself costs a request, and
#: it has to happen while there is still room to make one.
DEFAULT_THRESHOLD_RATIO = 0.8

#: Fraction of the window kept verbatim as the recent tail. dsh's default. The
#: most recent exchanges are what the next answer depends on, so they are never
#: summarized -- compaction eats the beginning, not the end.
DEFAULT_RETAIN_RATIO = 0.16

#: Marker left where a pruned tool result's middle was.
PRUNE_MARKER = "\n... [{removed} characters removed by compaction] ...\n"

#: Tool results longer than this are candidates for pruning.
DEFAULT_PRUNE_THRESHOLD = 4_000

#: Characters kept from each end of a pruned result. The head carries what the
#: tool was asked and how it started; the tail carries how it ended. The middle
#: of a long file listing is the part nobody refers back to.
DEFAULT_PRUNE_HEAD = 1_200
DEFAULT_PRUNE_TAIL = 600


class Summarizer(Protocol):
    """Writes a summary of a compacted range."""

    async def summarize(self, transcript: str) -> str: ...


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """What one compaction did."""

    compaction_id: str
    kind: str
    shadowed_seqs: tuple[int, ...] = ()
    shadowed_tokens: int = 0
    summary: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.shadowed_seqs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compaction": self.compaction_id,
            "kind": self.kind,
            "shadowed": len(self.shadowed_seqs),
            "shadowed_tokens": self.shadowed_tokens,
            "summary": self.summary,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# tool pairing
# --------------------------------------------------------------------------


def tool_call_balance(session: Session) -> list[bool]:
    """Whether each cut in the current surface is tool-pairing balanced.

    A surface of N nodes has N+1 cuts; entry `i` is the cut *before* node `i`
    and the last entry is the cut after the tail. An assistant message opens as
    many calls as it made and each result closes one, so a cut is balanced
    exactly where nothing is outstanding.

    Derived from content rather than from step markers because compaction moves
    surface positions: after one replacement, "where step 4 began" no longer
    names a place in the surface.
    """
    balanced = [True]
    outstanding = 0
    for event in session.surface_events():
        if event.type == ASSISTANT_MESSAGE:
            outstanding += len(event.data.get("calls") or [])
        elif event.type == TOOL_RESULT:
            outstanding -= 1
        if outstanding < 0:
            # A result with no open call: the surface is malformed, and
            # compacting it would turn a corrupt log into a corrupt request.
            raise ValueError(f"tool pairing: result at seq {event.seq} answers no call")
        balanced.append(outstanding == 0)
    return balanced


# --------------------------------------------------------------------------
# region selection
# --------------------------------------------------------------------------


def select_compactable_range(
    session: Session, measurement: Measurement, retain_tokens: int
) -> tuple[int, int] | None:
    """The inclusive span of surface seqs to compact, or None.

    Three rules, in order:

    1. The system prompt at the head is never inside the range. It is the
       instruction the whole session runs under, and summarizing it would
       change the agent rather than shorten the conversation.
    2. A recent tail worth `retain_tokens` is kept verbatim, because the most
       recent exchanges are what the next answer depends on.
    3. The cut moves *earlier* until it is tool-pairing balanced, so a call and
       its result are never separated.

    Returns None when applying all three leaves nothing worth compacting, which
    is the ordinary answer early in a session.
    """
    nodes = measurement.nodes
    if not nodes:
        return None

    first = 1 if nodes[0].type == SYSTEM_MESSAGE else 0

    # Walk back from the tail until the retained budget is met.
    accumulated = 0
    keep_from = len(nodes)
    for index in range(len(nodes) - 1, -1, -1):
        accumulated += nodes[index].tokens
        keep_from = index
        if accumulated >= retain_tokens:
            break
    if keep_from <= first:
        return None

    balanced = tool_call_balance(session)
    while keep_from > first and not balanced[keep_from]:
        keep_from -= 1
    if keep_from <= first:
        return None

    return nodes[first].seq, nodes[keep_from - 1].seq


def render_transcript(session: Session, seqs: list[int]) -> str:
    """The shadowed range as text for a summarizer to read."""
    by_seq = {event.seq: event for event in session.events}
    lines: list[str] = []
    for seq in seqs:
        event = by_seq.get(seq)
        if event is None:
            continue
        data = event.data
        if event.type == USER_MESSAGE:
            lines.append(f"User: {data.get('text', '')}")
        elif event.type == ASSISTANT_MESSAGE:
            text = str(data.get("text", ""))
            if text:
                lines.append(f"Assistant: {text}")
            for name in data.get("calls") or []:
                lines.append(f"Assistant called: {name}")
        elif event.type == TOOL_RESULT:
            content = str(data.get("content", ""))
            trimmed = content if len(content) <= 500 else content[:500] + " ..."
            lines.append(f"Tool {data.get('name', '')} returned: {trimmed}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the transaction
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Compactor:
    """Runs compactions over one session."""

    meter: TokenMeter = field(default_factory=TokenMeter)
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    retain_ratio: float = DEFAULT_RETAIN_RATIO
    prune_threshold: int = DEFAULT_PRUNE_THRESHOLD
    prune_head: int = DEFAULT_PRUNE_HEAD
    prune_tail: int = DEFAULT_PRUNE_TAIL

    def should_compact(self, measurement: Measurement) -> bool:
        """Whether pressure has reached the threshold."""
        if not measurement.context_window:
            return False
        return measurement.pressure >= self.threshold_ratio

    def retain_tokens(self) -> int:
        return int(self.meter.context_window * self.retain_ratio)

    def unmatched_start(self, session: Session) -> bool:
        """Whether a previous compaction crashed between its brackets.

        Deliberately reported rather than repaired. An unmatched start means
        a summarizer died mid-transaction, and the session should not quietly
        begin a second compaction on top of the first one's wreckage.
        """
        return len(session.of_type(COMPACTION_START)) > len(session.of_type(COMPACTION_END))

    # -- pruning: model-free, deterministic --------------------------------

    def prune(self, session: Session) -> CompactionResult:
        """Replace the middles of oversized tool results.

        Tried before summarizing because it costs nothing. A 40,000-character
        file read is almost never needed in full three turns later, and
        removing its middle often relieves the pressure without a request.

        Head and tail are kept: the head says what the tool returned and how it
        began, the tail says how it ended. The middle of a long listing is the
        part nobody refers back to.
        """
        compaction_id = f"c_{uuid.uuid4().hex[:10]}"
        candidates = [
            event
            for event in session.surface_events()
            if event.type == TOOL_RESULT
            and len(str(event.data.get("content", ""))) > self.prune_threshold
        ]
        if not candidates:
            return CompactionResult(compaction_id=compaction_id, kind="prune")

        session.append(COMPACTION_START, {"compaction": compaction_id, "kind": "prune"})
        shadowed: list[int] = []
        saved = 0

        for event in candidates:
            content = str(event.data.get("content", ""))
            removed = len(content) - self.prune_head - self.prune_tail
            pruned = (
                content[: self.prune_head]
                + PRUNE_MARKER.format(removed=removed)
                + content[-self.prune_tail :]
            )
            before = estimate_text(content)
            after = estimate_text(pruned)
            saved += before - after
            shadowed.append(event.seq)

            # Priced immediately before the replacement it prices. dsh calls
            # this the shadow-price protocol, and the adjacency is the
            # contract: a consumer subtracts the price it finds directly
            # before a replacement rather than retaining per-node prices.
            session.append(
                COMPACTION_PRUNE,
                {
                    "compaction": compaction_id,
                    "shadowed_seqs": [event.seq],
                    "shadowed_tokens": before - after,
                },
            )
            # The replacement carries what it shadows. Written on the
            # replacement rather than on the metering event so the surface fold
            # understands one kind of thing rather than two -- and written at
            # append time, because an event is a fact and editing one after the
            # fact is the thing this whole design exists to avoid.
            session.append(
                TOOL_RESULT,
                {
                    **event.data,
                    "content": pruned,
                    "pruned": True,
                    "shadowed_seqs": [event.seq],
                },
                sources=[event.seq],
            )

        session.append(COMPACTION_END, {"compaction": compaction_id})
        return CompactionResult(
            compaction_id=compaction_id,
            kind="prune",
            shadowed_seqs=tuple(shadowed),
            shadowed_tokens=saved,
        )

    # -- summarizing -------------------------------------------------------

    async def compact(
        self, session: Session, summarizer: Summarizer, *, schemas: Any = None
    ) -> CompactionResult:
        """Summarize the compactable range, if there is one.

        The whole transaction sits inside the bracket, model call included, so
        a failure leaves an unmatched start rather than a half-replaced
        surface.
        """
        compaction_id = f"c_{uuid.uuid4().hex[:10]}"

        if self.unmatched_start(session):
            return CompactionResult(
                compaction_id=compaction_id,
                kind="summary",
                error="a previous compaction did not finish",
            )

        measurement = self.meter.measure(session, schemas=schemas)
        span = select_compactable_range(session, measurement, self.retain_tokens())
        if span is None:
            return CompactionResult(compaction_id=compaction_id, kind="summary")

        start, end = span
        surface = session.surface()
        try:
            first_index = surface.index(start)
            last_index = surface.index(end)
        except ValueError:
            return CompactionResult(
                compaction_id=compaction_id, kind="summary", error="range left the surface"
            )

        shadowed = surface[first_index : last_index + 1]
        priced = {node.seq: node.tokens for node in measurement.nodes}
        shadowed_tokens = sum(priced.get(seq, 0) for seq in shadowed)

        session.append(
            COMPACTION_START,
            {"compaction": compaction_id, "kind": "summary", "range": [start, end]},
        )
        try:
            summary = await summarizer.summarize(render_transcript(session, shadowed))
        except Exception as exc:
            logger.exception("compaction %s: summarizer failed", compaction_id)
            session.append(COMPACTION_END, {"compaction": compaction_id, "error": str(exc)})
            return CompactionResult(compaction_id=compaction_id, kind="summary", error=str(exc))

        if not summary.strip():
            # An empty summary would shadow the conversation with nothing,
            # which is worse than not compacting: the model would lose the
            # history and gain no account of it.
            session.append(COMPACTION_END, {"compaction": compaction_id, "error": "empty summary"})
            return CompactionResult(
                compaction_id=compaction_id, kind="summary", error="empty summary"
            )

        session.append(
            COMPACTION_SUMMARY,
            {
                "compaction": compaction_id,
                "summary": summary,
                "shadowed_seqs": list(shadowed),
                "shadowed_tokens": shadowed_tokens,
                "range": [start, end],
            },
        )
        # The replacement, immediately after its price. A user message rather
        # than a system one: it is content the model reads as part of the
        # conversation, and putting it in the prompt would make every later
        # compaction rewrite the instructions.
        session.append(
            USER_MESSAGE,
            {
                "text": summary,
                "kind": "summary",
                "compaction": compaction_id,
                "shadowed_seqs": list(shadowed),
            },
        )
        session.append(COMPACTION_END, {"compaction": compaction_id})

        return CompactionResult(
            compaction_id=compaction_id,
            kind="summary",
            shadowed_seqs=tuple(shadowed),
            shadowed_tokens=shadowed_tokens,
            summary=summary,
        )


#: What the summarizer is told to produce. Written for the thing that reads it
#: next, which is a model continuing a conversation -- not a person skimming.
#: Decisions and open threads survive; pleasantries do not.
SUMMARY_PROMPT = """You are compacting the earlier part of a conversation so it \
fits in a context window. Write a dense summary that another model will read as \
its only record of what happened.

Keep: decisions reached, facts established, file paths and identifiers, what was \
tried and failed, and anything still outstanding.
Drop: pleasantries, restated questions, and any detail that changed later.

Write in the language the conversation is in. Do not address the user."""


@dataclass(slots=True)
class ModelSummarizer:
    """Summarizes by asking the model, through koe's adapter seam."""

    adapter: Any
    max_tokens: int = 8_192

    async def summarize(self, transcript: str) -> str:
        from koe.agent.loop import ChatMessage
        from koe.text.thinking import strip_thinking

        reply = await self.adapter.reply(
            [ChatMessage(role="user", content=transcript)],
            system=SUMMARY_PROMPT,
            tools=[],
        )
        return strip_thinking(reply.text).strip()
