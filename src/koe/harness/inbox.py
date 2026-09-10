"""Pending input: what the agent will be told, and when.

Ported from DeepSeek Harness's `ReactLoopInbox` (MIT). It is the piece koe's
chat had no equivalent of, and its absence is why the old chat could only be
talked to between answers.

**Two queues, because "say this next" and "say this now" are different
requests.**

``next_turn`` holds prompts that each get a turn of their own. This is the
ordinary case: you type, the agent answers, you type again.

``next_step`` holds input that joins the turn *already running*, at its next
step boundary. This is steering, and it is the interesting one. A model three
tool calls into the wrong file does not need to be cancelled and re-prompted --
it needs to be told, while it is working, that it is looking in the wrong place.
The correction arrives at the next boundary and the turn continues with it in
history, so the work already done is kept.

**Claiming is atomic and durable.** A boundary takes the whole `next_step`
batch, plus at most one queued turn, in a single operation recorded in the
session log. Two things follow. The agent cannot half-consume a batch, so a
crash between claiming and using cannot lose a message; and the queue's history
replays, so what the model was told and when is reconstructible rather than
remembered.

**Every mutation is a splice event, not a mutation.** Append, cancel, claim and
clear all become one `agent/inbox/spliced` fact. The queue is a fold over those
facts, exactly as the message history is a fold over the message ones -- which
means the same guarantee: the state can be rebuilt, and the reason it holds
what it holds is in the log.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from koe.harness.session import INBOX_SPLICED, Session

#: Which queue a message is bound for.
InboxTarget = Literal["next-turn", "next-step"]

NEXT_TURN: InboxTarget = "next-turn"
NEXT_STEP: InboxTarget = "next-step"


@dataclass(frozen=True, slots=True)
class Pending:
    """One queued message.

    `kind` separates what a person typed from what the harness injected -- a
    tool's extra context, a system notice. They read identically to the model
    and must not read identically in the UI, or the transcript will show the
    user saying things they never said.
    """

    text: str
    kind: Literal["user", "context"] = "user"
    id: str = field(default_factory=lambda: f"m_{uuid.uuid4().hex[:10]}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "kind": self.kind}


def _from_dict(raw: dict[str, Any]) -> Pending:
    return Pending(
        text=str(raw.get("text", "")),
        kind="context" if raw.get("kind") == "context" else "user",
        id=str(raw.get("id") or f"m_{uuid.uuid4().hex[:10]}"),
    )


class Inbox:
    """Durable pending input for one agent.

    State is a fold over the session's splice events, so it is never stored
    twice: the log is the queue.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- reading ----------------------------------------------------------

    def _state(self) -> dict[InboxTarget, list[Pending]]:
        """Rebuild both queues by replaying every splice."""
        state: dict[InboxTarget, list[Pending]] = {NEXT_TURN: [], NEXT_STEP: []}
        for event in self._session.of_type(INBOX_SPLICED):
            target: InboxTarget = NEXT_STEP if event.data.get("target") == NEXT_STEP else NEXT_TURN
            queue = state[target]
            start = int(event.data.get("start", 0))
            removed = int(event.data.get("removed", 0))
            inserted = [_from_dict(raw) for raw in event.data.get("inserted") or []]
            if not (0 <= start <= len(queue) and removed >= 0 and start + removed <= len(queue)):
                # A log this malformed cannot be folded into a queue, and
                # guessing would put the agent to work on invented input.
                raise ValueError(f"invalid inbox splice at seq {event.seq}")
            state[target] = queue[:start] + inserted + queue[start + removed :]
        return state

    @property
    def next_turn(self) -> list[Pending]:
        return self._state()[NEXT_TURN]

    @property
    def next_step(self) -> list[Pending]:
        return self._state()[NEXT_STEP]

    @property
    def pending(self) -> bool:
        state = self._state()
        return bool(state[NEXT_TURN] or state[NEXT_STEP])

    def __len__(self) -> int:
        state = self._state()
        return len(state[NEXT_TURN]) + len(state[NEXT_STEP])

    # -- writing ----------------------------------------------------------

    def _splice(
        self,
        target: InboxTarget,
        start: int,
        removed: int,
        inserted: Sequence[Pending] = (),
    ) -> list[Pending]:
        """Record one splice and return what it removed."""
        before = self._state()[target]
        taken = before[start : start + removed]
        self._session.append(
            INBOX_SPLICED,
            {
                "target": target,
                "start": start,
                "removed": removed,
                "inserted": [message.to_dict() for message in inserted],
            },
        )
        return taken

    def send(self, message: Pending | str, target: InboxTarget = NEXT_TURN) -> Pending:
        """Queue one message."""
        entry = Pending(text=message) if isinstance(message, str) else message
        queue = self._state()[target]
        self._splice(target, len(queue), 0, [entry])
        return entry

    def followup(self, text: str) -> Pending:
        """Queue a prompt that gets its own turn. The ordinary case."""
        return self.send(Pending(text=text), NEXT_TURN)

    def steer(self, text: str) -> Pending:
        """Join the running turn at its next step boundary.

        The turn keeps the work it has already done. That is the difference
        between steering and cancelling, and it is the reason both exist.
        """
        return self.send(Pending(text=text), NEXT_STEP)

    def inject(self, text: str) -> Pending:
        """Add harness-authored context to the running turn.

        Same queue as steering, different `kind`: a tool that produces context
        for the next step is not the user speaking, and a UI that renders it as
        though it were will show a transcript of things nobody said.
        """
        return self.send(Pending(text=text, kind="context"), NEXT_STEP)

    def claim(self, target: InboxTarget, turn: int) -> list[Pending]:
        """Take the batch for one boundary, atomically.

        Always the whole of `next_step`; plus at most one queued turn when the
        boundary is a turn boundary. One queue is drained and the other is
        sipped, because a step boundary is a continuation of a turn and a turn
        boundary is the start of a new one -- taking two prompts at a turn
        boundary would silently merge two things the user asked separately.
        """
        claimed = self._splice(NEXT_STEP, 0, len(self._state()[NEXT_STEP]))
        if target == NEXT_TURN:
            queued = self._state()[NEXT_TURN]
            if queued:
                claimed.extend(self._splice(NEXT_TURN, 0, 1))
        return claimed

    def clear(self) -> list[Pending]:
        """Cancel everything pending.

        `next_step` before `next_turn`, so a fold that is interrupted between
        the two splices leaves the queue that has not been started yet rather
        than the one that has.
        """
        dropped = self._splice(NEXT_STEP, 0, len(self._state()[NEXT_STEP]))
        dropped.extend(self._splice(NEXT_TURN, 0, len(self._state()[NEXT_TURN])))
        return dropped

    def to_dict(self) -> dict[str, Any]:
        state = self._state()
        return {
            "next_turn": [message.to_dict() for message in state[NEXT_TURN]],
            "next_step": [message.to_dict() for message in state[NEXT_STEP]],
        }
