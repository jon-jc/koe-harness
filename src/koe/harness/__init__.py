"""koe harness: the agent runtime behind the chat.

A port of the parts of [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(MIT) that make a chat surface a *harness* rather than a chat box. dsh is a
TypeScript monorepo on Cordis and koe's backend is Python, so this is a port and
not a vendoring: the architecture, the event taxonomy and the scheduling
algorithm are dsh's, reimplemented against koe's kernel, tool registry and
providers.

Four things arrive together, and they are one design rather than four features.

**A session is an append-only log, and the message history is derived from it.**
Not a mutated list. That single change is what makes every other one possible:
the inbox is a fold over the same log, cancellation has somewhere to record the
prefix the user actually saw, and "what did the model see on step 3" has an
answer.

**Input is queued, not called.** `send()` puts a message on a queue and wakes a
driver. A message arriving mid-turn is therefore ordinary rather than a
concurrency hazard, which is what makes **steering** possible -- correcting a
model that is three tool calls into the wrong file, without cancelling the work
it has already done.

**Tool calls run in parallel where the tool says that is safe, and commit in
model order.** Dispatch overlaps; the log does not. A read that finishes first
still lands after the calls the model listed before it, so the derived history
does not depend on disk timing.

**Cancellation is explicit, cooperative, and leaves a well-formed history.**
Streamed text is committed as interrupted; calls that never dispatched get a
synthetic result, because an assistant turn containing a call with no result is
a request vendors reject.

What is deliberately absent: dsh's compaction, projections registry, persistence
backends, delegation and ACP. Those are real parts of dsh and koe does not have
them -- said plainly here rather than stubbed, so nobody reads this package as a
complete implementation of that one.
"""

from koe.harness.agent import (
    MAX_STEPS,
    AgentHandle,
    AgentRegistry,
    HarnessAgent,
    TurnOutcome,
    TurnReason,
)
from koe.harness.inbox import NEXT_STEP, NEXT_TURN, Inbox, InboxTarget, Pending
from koe.harness.scheduler import (
    DEFAULT_MAX_PARALLEL,
    ToolBatch,
    execute_tool_calls,
    parse_arguments,
)
from koe.harness.session import Session, SessionEvent

__all__ = [
    "DEFAULT_MAX_PARALLEL",
    "MAX_STEPS",
    "NEXT_STEP",
    "NEXT_TURN",
    "AgentHandle",
    "AgentRegistry",
    "HarnessAgent",
    "Inbox",
    "InboxTarget",
    "Pending",
    "Session",
    "SessionEvent",
    "ToolBatch",
    "TurnOutcome",
    "TurnReason",
    "execute_tool_calls",
    "parse_arguments",
]
