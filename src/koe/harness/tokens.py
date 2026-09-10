"""Estimating how much of the context window a conversation is using.

Compaction needs a price per node before it can decide what to drop, and the
price has to be available *without* asking a provider — the decision is made
while assembling a request, and a round trip to count tokens for a request you
have not sent yet is absurd.

So this is a heuristic, and it says so. dsh keeps its estimator behind a
`TokenMeter` seam for the same reason: the number is used for a *relative*
judgement (which nodes are big, are we near the limit) where being consistently
20% out costs a slightly early compaction, and being wrong about the ratio
between two nodes costs the wrong choice.

**The ratio is where a naive estimator goes badly wrong in Japanese**, which is
the whole reason this file is not `len(text) // 4`.

English averages roughly four characters per token: byte-pair encoders learned
whole English words, so ``information`` is one or two tokens. Japanese averages
closer to *one* token per character, and often worse — most kanji are their own
token, and many are two or three, because a UTF-8 kanji is three bytes and the
vocabulary was not built for them. A `len // 4` estimator therefore prices a
Japanese meeting transcript at a quarter of its real cost, and the harness
would sail past the context limit believing it had three times the room it has.

That is not a rounding error. It is the failure mode of an English-tuned
heuristic in the language koe exists for, so the estimator counts scripts
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from koe.harness.session import (
    ASSISTANT_MESSAGE,
    SYSTEM_MESSAGE,
    TOOL_CALL,
    TOOL_RESULT,
    USER_MESSAGE,
    Session,
    SessionEvent,
)
from koe.text.script import classify

#: Characters per token for Latin text. Byte-pair encoders learned English
#: words, so this is generous and stable across providers.
LATIN_CHARS_PER_TOKEN = 4.0

#: Tokens per character for Japanese. Above one because a kanji is three bytes
#: of UTF-8 and the vocabulary was not built around them, so many cost two or
#: three tokens on their own. 1.1 is deliberately conservative: for a decision
#: about whether to compact, over-estimating costs one early compaction and
#: under-estimating costs a request the provider refuses.
JA_TOKENS_PER_CHAR = 1.1

#: Every message carries role markers and delimiters the text does not show.
PER_MESSAGE_OVERHEAD = 4

#: A tool call carries its name and a JSON envelope around the arguments.
PER_TOOL_CALL_OVERHEAD = 10


def estimate_text(text: str) -> int:
    """Tokens for one string, counting scripts separately.

    One pass, classifying each character. Japanese and Latin are counted with
    their own ratios and everything else -- digits, punctuation, whitespace --
    is priced as Latin, which is what an encoder does with them anyway.
    """
    if not text:
        return 0

    japanese = 0
    other = 0
    for char in text:
        if classify(char).is_japanese:
            japanese += 1
        else:
            other += 1

    return int(japanese * JA_TOKENS_PER_CHAR + other / LATIN_CHARS_PER_TOKEN) + 1


def estimate_event(event: SessionEvent) -> int:
    """Tokens one surface node contributes to a request."""
    data = event.data
    if event.type in (USER_MESSAGE, ASSISTANT_MESSAGE, SYSTEM_MESSAGE):
        return estimate_text(str(data.get("text", ""))) + PER_MESSAGE_OVERHEAD
    if event.type == TOOL_CALL:
        arguments = data.get("arguments") or {}
        rendered = " ".join(f"{key}={value}" for key, value in dict(arguments).items())
        return (
            estimate_text(str(data.get("name", "")))
            + estimate_text(rendered)
            + PER_TOOL_CALL_OVERHEAD
        )
    if event.type == TOOL_RESULT:
        return estimate_text(str(data.get("content", ""))) + PER_MESSAGE_OVERHEAD
    return 0


@dataclass(frozen=True, slots=True)
class NodePrice:
    """One surface node and what it costs."""

    seq: int
    type: str
    tokens: int


@dataclass(frozen=True, slots=True)
class Measurement:
    """The priced surface, and the pressure it puts on the window."""

    nodes: tuple[NodePrice, ...] = ()
    #: What the tool schemas cost. Paid on every step and easy to forget: a
    #: dozen tools with real descriptions is a four-figure constant.
    schema_tokens: int = 0
    context_window: int = 0

    @property
    def message_tokens(self) -> int:
        return sum(node.tokens for node in self.nodes)

    @property
    def total(self) -> int:
        return self.message_tokens + self.schema_tokens

    @property
    def pressure(self) -> float:
        """Fraction of the window in use. 0 when the window is unknown."""
        return self.total / self.context_window if self.context_window else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_tokens": self.message_tokens,
            "schema_tokens": self.schema_tokens,
            "total": self.total,
            "context_window": self.context_window,
            "pressure": round(self.pressure, 4),
            "nodes": len(self.nodes),
        }


@dataclass(slots=True)
class TokenMeter:
    """Prices a session's surface.

    A class rather than a function so a deployment can substitute a real
    tokenizer -- `tiktoken`, a provider's counting endpoint -- without anything
    that consumes a measurement changing. dsh keeps the same seam, and for the
    same reason: the estimate is load-bearing for a decision, so it must be
    replaceable by a measurement.
    """

    context_window: int = 128_000
    _schema_cache: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def measure(
        self, session: Session, *, schemas: list[dict[str, Any]] | None = None
    ) -> Measurement:
        nodes = tuple(
            NodePrice(seq=event.seq, type=event.type, tokens=estimate_event(event))
            for event in session.surface_events()
        )
        return Measurement(
            nodes=nodes,
            schema_tokens=self.measure_schemas(schemas or []),
            context_window=self.context_window,
        )

    def measure_schemas(self, schemas: list[dict[str, Any]]) -> int:
        """What the tool definitions cost, cached on their shape.

        Schemas change rarely and are re-priced on every step otherwise, which
        is a full walk of every description and parameter object per model
        request for an answer that was the same last time.
        """
        if not schemas:
            return 0
        key = hash(tuple(sorted(schema.get("name", "") for schema in schemas)))
        cached = self._schema_cache.get(key)
        if cached is not None:
            return cached

        total = 0
        for schema in schemas:
            total += estimate_text(str(schema.get("name", "")))
            total += estimate_text(str(schema.get("description", "")))
            total += estimate_text(str(schema.get("parameters", "")))
        self._schema_cache[key] = total
        return total
