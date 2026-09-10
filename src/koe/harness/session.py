"""The durable session log.

Ported from DeepSeek Harness (MIT), whose central architectural claim is the
one worth taking: **a conversation is not a list of messages, it is an
append-only log of facts, and the messages are derived from it.**

koe's chat used to hold a `list[ChatMessage]` and mutate it. That works until
you ask it anything:

* *What did the model actually see on step 3?* A mutated list cannot say. The
  log can, because every request's inputs are events that were already written.
* *A tool call was refused — when, and by what?* In a message list the refusal
  is either lost or flattened into prose. Here it is an event with a code.
* *The user cancelled mid-stream. What now?* The delivered prefix is already
  in the log as an `assistant/message` marked interrupted, so the next request
  contains what the user actually saw rather than either nothing or a
  completion they never read.
* *Replay it.* Deriving from an append-only log is reproducible. Replaying
  mutations is not.

**Events are facts, and facts are not retracted.** Nothing here edits or
removes an event. A correction is a later event that shadows an earlier one,
which is why `derive_messages` is a fold rather than a filter: the fold is where
"what the model sees now" is decided, and it can change without the history
changing.

**Sequence numbers are dense and assigned here.** A tool result cites the seq of
the call it answers, which is what lets a UI pair them without matching on ids
that a model chose.

The taxonomy is dsh's. koe's fold is smaller: dsh supports compaction,
projections, and multi-agent delegation that koe does not, and inventing the
event names would have made the two impossible to compare.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from koe.agent.loop import ChatMessage, ToolCall
from koe.tools.registry import ToolError, ToolResult

# -- the taxonomy ----------------------------------------------------------
# String constants rather than an enum: these are wire values that a plugin
# subscribes to and a stored log replays, so their spelling is the contract.

SESSION_CREATED = "session/created"
SYSTEM_MESSAGE = "system/message"
USER_MESSAGE = "user/message"
ASSISTANT_MESSAGE = "assistant/message"
TOOL_CALL = "tool/call"
TOOL_RESULT = "tool/result"
TURN_START = "turn/start"
TURN_END = "turn/end"
STEP_START = "step/start"
STEP_END = "step/end"
INBOX_SPLICED = "agent/inbox/spliced"
REQUEST_HEADER = "request/header"

#: Compaction. The bracket is a lock: an unmatched `compaction/start` in a log
#: is a compaction that crashed, and it is meant to stay detectable.
COMPACTION_START = "compaction/start"
COMPACTION_SUMMARY = "compaction/summary"
COMPACTION_PRUNE = "compaction/prune"
COMPACTION_END = "compaction/end"
#: Shadows a range of surface nodes. The replacement itself is an ordinary
#: message event carrying this in its data, so the thing that replaces history
#: is history.
SURFACE_REPLACE = "surface/replace"

#: Events that contribute to what the model sees. Everything else is
#: bookkeeping: turn boundaries, headers, inbox splices. Keeping the list
#: explicit is what stops a new event type silently entering the prompt.
DERIVING = frozenset({SYSTEM_MESSAGE, USER_MESSAGE, ASSISTANT_MESSAGE, TOOL_CALL, TOOL_RESULT})


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One fact, at one position in the log."""

    seq: int
    type: str
    data: dict[str, Any]
    at: float = field(default_factory=time.time)
    #: The events this one answers. A tool result cites its call, so a UI pairs
    #: them without matching on an id the model chose.
    sources: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "data": self.data,
            "at": self.at,
            "sources": list(self.sources),
        }


@dataclass(slots=True)
class Session:
    """An append-only log, and the message history derived from it."""

    id: str = field(default_factory=lambda: f"s_{uuid.uuid4().hex[:12]}")
    system: str = ""
    _events: list[SessionEvent] = field(default_factory=list, init=False, repr=False)
    _seq: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.append(SESSION_CREATED, {"session": self.id})
        if self.system:
            self.append(SYSTEM_MESSAGE, {"text": self.system})

    # -- writing ----------------------------------------------------------

    def append(
        self, type: str, data: dict[str, Any], *, sources: Sequence[int] = ()
    ) -> SessionEvent:
        """Record one fact. Returns the event, whose `seq` others may cite."""
        if self._closed:
            raise RuntimeError(f"session {self.id} is closed")
        self._seq += 1
        event = SessionEvent(seq=self._seq, type=type, data=dict(data), sources=tuple(sources))
        self._events.append(event)
        return event

    def close(self) -> None:
        """Stop accepting writes. Reading and deriving still work."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # -- reading ----------------------------------------------------------

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[SessionEvent]:
        return iter(self._events)

    def surface(self) -> list[int]:
        """The seqs a model request would carry, in order.

        The log is everything that happened; the **surface** is what is still
        visible. Compaction does not delete -- it appends a replacement that
        *shadows* a range, so the summary and the events it stands for both
        remain, and "what was compacted away" has an answer.

        Order matters and is subtle: a replacement lands at the *position of
        the range it shadows*, not at the end. A summary written at seq 900 for
        a range starting at seq 4 appears where seq 4 was, because that is
        where the conversation it summarizes happened. Appending it at the end
        would reorder the conversation.
        """
        shadowed: set[int] = set()
        replacements: dict[int, list[int]] = {}
        for event in self._events:
            hidden = event.data.get("shadowed_seqs")
            if event.type == SURFACE_REPLACE or (hidden and isinstance(hidden, list)):
                seqs = [int(seq) for seq in hidden or []]
                if not seqs:
                    continue
                shadowed.update(seqs)
                # Anchored at the earliest node it replaces.
                replacements.setdefault(min(seqs), []).append(event.seq)

        visible: list[int] = []
        for event in self._events:
            if event.seq in shadowed:
                # The replacement takes the position of the first node it
                # shadows, so the summary reads where the conversation was.
                for replacement in replacements.get(event.seq, ()):
                    if replacement not in shadowed:
                        visible.append(replacement)
                continue
            if event.type not in DERIVING:
                continue
            if event.data.get("shadowed_seqs"):
                # Already placed at its anchor above.
                continue
            visible.append(event.seq)
        return visible

    def surface_events(self) -> list[SessionEvent]:
        """The surface as events, in surface order."""
        by_seq = {event.seq: event for event in self._events}
        return [by_seq[seq] for seq in self.surface() if seq in by_seq]

    def of_type(self, *types: str) -> list[SessionEvent]:
        wanted = frozenset(types)
        return [event for event in self._events if event.type in wanted]

    def since(self, seq: int) -> list[SessionEvent]:
        """Events after `seq`, for a client catching up on a live session."""
        return [event for event in self._events if event.seq > seq]

    @property
    def last_seq(self) -> int:
        return self._seq

    def request_header(self) -> dict[str, Any] | None:
        """The most recent request envelope, or None before the first request.

        Used to decide whether the next request needs a fresh header: an
        unchanged envelope inherits the last one, which is what keeps a
        provider's prefix cache usable across steps.
        """
        headers = self.of_type(REQUEST_HEADER)
        return headers[-1].data if headers else None

    # -- deriving ---------------------------------------------------------

    def system_prompt(self) -> str:
        """The effective prompt: the latest non-empty `system/message`.

        Latest rather than first, and non-empty rather than last, because a
        prompt is *replaced* by appending a new one and *cleared* by appending
        an empty one. Both operations are then visible in history instead of
        editing a fact already written.
        """
        text = ""
        for event in self._events:
            if event.type is SYSTEM_MESSAGE or event.type == SYSTEM_MESSAGE:
                candidate = str(event.data.get("text") or "")
                text = candidate
        return text

    def derive_messages(self) -> list[ChatMessage]:
        """Fold the log into the history a model request carries.

        The fold is the whole point. Tool calls and their results live as
        separate events -- a call is a fact the moment the model asks, whether
        or not it is ever answered -- and only here are they gathered back into
        the assistant/tool message pair a vendor expects.
        """
        messages: list[ChatMessage] = []
        pending_results: list[ToolResult] = []
        #: The assistant message that later `tool/call` events belong to. The
        #: anchor is written before the calls it made -- the step commits the
        #: text first so a cancellation cannot lose it -- so calls attach
        #: backwards, to the message already emitted, not forwards to the next.
        anchor: ChatMessage | None = None

        def flush_results() -> None:
            if pending_results:
                messages.append(ChatMessage(role="tool", tool_results=list(pending_results)))
                pending_results.clear()

        for event in self.surface_events():
            if event.type not in DERIVING or event.type == SYSTEM_MESSAGE:
                # The prompt is carried on the request as a field rather than as
                # history: see `system_prompt`. Skipped so it cannot appear
                # twice.
                continue
            data = event.data

            if event.type == USER_MESSAGE:
                flush_results()
                anchor = None
                messages.append(ChatMessage(role="user", content=str(data.get("text", ""))))
            elif event.type == ASSISTANT_MESSAGE:
                flush_results()
                anchor = ChatMessage(role="assistant", content=str(data.get("text", "")))
                messages.append(anchor)
            elif event.type == TOOL_CALL:
                if anchor is None:
                    # A call with no anchor: the log was cut between the model
                    # asking and the text being committed. One is synthesized so
                    # the history stays well-formed rather than dropping the
                    # call, which would leave its result answering nothing.
                    anchor = ChatMessage(role="assistant", content="")
                    messages.append(anchor)
                anchor.tool_calls.append(
                    ToolCall(
                        id=str(data.get("call_id", "")),
                        name=str(data.get("name", "")),
                        arguments=dict(data.get("arguments") or {}),
                    )
                )
            elif event.type == TOOL_RESULT:
                pending_results.append(_result_from(data))

        flush_results()
        return messages

    # -- introspection ----------------------------------------------------

    def turns(self) -> list[dict[str, Any]]:
        """Turn and step structure, for the harness view.

        Derived rather than tracked: the log already says where every boundary
        is, and a second copy of that would be a second thing to keep correct.
        """
        out: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for event in self._events:
            if event.type == TURN_START:
                current = {"turn": event.data.get("turn"), "steps": [], "reason": None}
                out.append(current)
            elif event.type == STEP_START and current is not None:
                current["steps"].append({"step": event.data.get("step"), "tools": []})
            elif event.type == TOOL_CALL and current is not None and current["steps"]:
                current["steps"][-1]["tools"].append(event.data.get("name"))
            elif event.type == TURN_END and current is not None:
                current["reason"] = event.data.get("reason")
                current = None
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.id,
            "last_seq": self._seq,
            "events": [event.to_dict() for event in self._events],
        }


def _result_from(data: dict[str, Any]) -> ToolResult:
    """Rebuild a ToolResult from its logged form."""
    raw_error = data.get("error")
    error: ToolError | None = None
    if raw_error:
        try:
            error = ToolError(raw_error)
        except ValueError:
            error = ToolError.FAILED
    return ToolResult(
        tool=str(data.get("name", "")),
        call_id=str(data.get("call_id", "")),
        ok=bool(data.get("ok", False)),
        content=str(data.get("content", "")),
        error=error,
        detail=str(data.get("detail", "")),
        duration_ms=float(data.get("duration_ms", 0.0)),
    )


def log_tool_result(
    session: Session, result: ToolResult, *, turn: int, step: int, call_seq: int
) -> SessionEvent:
    """Append one tool result, citing the call event it answers."""
    return session.append(
        TOOL_RESULT,
        {
            "turn": turn,
            "step": step,
            "call_id": result.call_id,
            "name": result.tool,
            "ok": result.ok,
            "content": result.content,
            "error": result.error.value if result.error else None,
            "detail": result.detail,
            "duration_ms": result.duration_ms,
        },
        sources=[call_seq],
    )
