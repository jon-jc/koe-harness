"""Compaction: the surface, token pricing, safe cuts, and the transaction."""

from __future__ import annotations

from typing import Any

import pytest

from koe.agent.loop import Adapter, ModelReply
from koe.harness import HarnessAgent
from koe.harness.compaction import (
    Compactor,
    render_transcript,
    select_compactable_range,
    tool_call_balance,
)
from koe.harness.session import (
    ASSISTANT_MESSAGE,
    COMPACTION_END,
    COMPACTION_START,
    COMPACTION_SUMMARY,
    TOOL_RESULT,
    USER_MESSAGE,
    Session,
)
from koe.harness.tokens import TokenMeter, estimate_text
from koe.kernel.context import Context
from koe.tools.registry import ToolRegistry


class Summary:
    """A summarizer that does not need a model."""

    def __init__(self, text: str = "EARLIER: they agreed to ship.") -> None:
        self.text = text
        self.seen: list[str] = []

    async def summarize(self, transcript: str) -> str:
        self.seen.append(transcript)
        return self.text


class Failing:
    async def summarize(self, transcript: str) -> str:
        raise RuntimeError("summarizer exploded")


class Empty:
    async def summarize(self, transcript: str) -> str:
        return "   "


def conversation(turns: int, *, size: int = 40) -> Session:
    """A session with messages of a realistic length.

    Sized deliberately rather than minimally: a conversation of one-line
    messages has less content than the retained tail alone, so every selection
    correctly returns None and the test proves nothing about selection.
    """
    session = Session(system="be helpful")
    for index in range(turns):
        session.append(USER_MESSAGE, {"text": f"question {index}. " + "please explain. " * size})
        session.append(
            ASSISTANT_MESSAGE,
            {"text": f"answer {index}. " + "here is the explanation. " * size, "calls": []},
        )
    return session


# --------------------------------------------------------------------------
# token pricing
# --------------------------------------------------------------------------


def test_japanese_costs_far_more_per_character_than_english() -> None:
    """The reason this is not `len(text) // 4`.

    A byte-pair encoder learned English words, so `information` is a token or
    two. Most kanji are their own token and many are two or three, because a
    UTF-8 kanji is three bytes and the vocabulary was not built for them.
    """
    english = "The quarterly revenue review is scheduled for Friday afternoon."
    japanese = "第三四半期の売上レビューは金曜日の午後に予定されています。"

    en_ratio = len(english) / estimate_text(english)
    ja_ratio = len(japanese) / estimate_text(japanese)

    assert en_ratio > 3.0, "English should be several characters per token"
    assert ja_ratio < 1.5, "Japanese should be about one token per character"


def test_an_english_tuned_estimator_would_badly_underprice_japanese() -> None:
    """Measured, because the size of the error is the argument: a harness that
    believes it has three times the room it has will sail past the limit."""
    japanese = "本日の議題は第三四半期の売上レビューです。" * 5
    naive = len(japanese) // 4
    assert estimate_text(japanese) > naive * 3


def test_the_schema_price_is_counted_and_not_forgotten() -> None:
    """A dozen tools with real descriptions is a four-figure constant paid on
    every single step."""
    meter = TokenMeter(context_window=1000)
    schemas = [
        {"name": "read_file", "description": "Read a file " * 20, "parameters": {"a": 1}}
    ] * 6
    assert meter.measure_schemas(schemas) > 100


def test_pressure_is_zero_when_the_window_is_unknown() -> None:
    """Rather than dividing by zero or inventing a limit."""
    meter = TokenMeter(context_window=0)
    assert meter.measure(conversation(3)).pressure == 0.0


# --------------------------------------------------------------------------
# safe cuts
# --------------------------------------------------------------------------


def test_a_cut_between_a_call_and_its_result_is_unsafe() -> None:
    """The rule that makes compaction safe. Cutting there produces an assistant
    turn whose calls are answered by results the request no longer contains,
    which every vendor rejects."""
    session = Session()
    session.append(USER_MESSAGE, {"text": "go"})
    session.append(ASSISTANT_MESSAGE, {"text": "", "calls": ["read", "read"]})
    session.append(TOOL_RESULT, {"name": "read", "content": "a"})
    session.append(TOOL_RESULT, {"name": "read", "content": "b"})
    session.append(ASSISTANT_MESSAGE, {"text": "done", "calls": []})

    balance = tool_call_balance(session)

    # Cut 0 is before everything, cut 2 is after the assistant that opened two
    # calls, cut 3 sits between the two results.
    assert balance[0] is True
    assert balance[2] is False
    assert balance[3] is False
    assert balance[4] is True


def test_a_result_answering_no_call_is_refused() -> None:
    """Compacting a corrupt surface would turn it into a corrupt request."""
    session = Session()
    session.append(TOOL_RESULT, {"name": "read", "content": "orphan"})
    with pytest.raises(ValueError, match="answers no call"):
        tool_call_balance(session)


def test_the_cut_moves_earlier_until_it_is_balanced() -> None:
    session = Session()
    session.append(USER_MESSAGE, {"text": "x" * 400})
    session.append(ASSISTANT_MESSAGE, {"text": "", "calls": ["read"]})
    session.append(TOOL_RESULT, {"name": "read", "content": "y" * 400})
    session.append(USER_MESSAGE, {"text": "z" * 400})

    meter = TokenMeter(context_window=1000)
    span = select_compactable_range(session, meter.measure(session), retain_tokens=120)

    if span is not None:
        surface = session.surface()
        end_index = surface.index(span[1])
        assert tool_call_balance(session)[end_index + 1], "range ended on an unsafe cut"


# --------------------------------------------------------------------------
# region selection
# --------------------------------------------------------------------------


def test_the_system_prompt_is_never_compacted() -> None:
    """It is the instruction the session runs under. Summarizing it would
    change the agent rather than shorten the conversation."""
    session = conversation(12)
    meter = TokenMeter(context_window=2000)
    span = select_compactable_range(session, meter.measure(session), retain_tokens=200)

    assert span is not None
    system_seq = session.surface()[0]
    assert span[0] != system_seq


def test_the_recent_tail_is_kept_verbatim() -> None:
    """Compaction eats the beginning. The most recent exchanges are what the
    next answer depends on."""
    session = conversation(20)
    meter = TokenMeter(context_window=4000)
    measurement = meter.measure(session)
    span = select_compactable_range(session, measurement, retain_tokens=400)

    assert span is not None
    surface = session.surface()
    kept = surface[surface.index(span[1]) + 1 :]
    retained = sum(node.tokens for node in measurement.nodes if node.seq in set(kept))
    assert retained >= 400


def test_a_short_conversation_has_nothing_worth_compacting() -> None:
    session = conversation(2)
    meter = TokenMeter(context_window=100_000)
    assert select_compactable_range(session, meter.measure(session), 16_000) is None


# --------------------------------------------------------------------------
# the transaction
# --------------------------------------------------------------------------


async def test_compaction_shadows_rather_than_deletes() -> None:
    """The whole design. A bug in the summarizer costs a bad summary, not a
    lost conversation."""
    session = conversation(20)
    before = len(session.events)
    compactor = Compactor(meter=TokenMeter(context_window=4000))

    result = await compactor.compact(session, Summary())

    assert result.ok
    assert len(session.events) > before, "events were removed"
    assert len(session.surface()) < before, "the surface did not shrink"
    # Everything shadowed is still in the log.
    for seq in result.shadowed_seqs:
        assert any(event.seq == seq for event in session.events)


async def test_the_summary_lands_where_the_conversation_was() -> None:
    """Not at the end. A summary of the beginning appended last would put the
    beginning after the middle."""
    session = conversation(20)
    compactor = Compactor(meter=TokenMeter(context_window=4000))

    await compactor.compact(session, Summary("EARLIER: the start of it."))

    messages = session.derive_messages()
    assert messages[0].content.startswith("EARLIER:")


async def test_the_bracket_closes_on_success() -> None:
    session = conversation(20)
    compactor = Compactor(meter=TokenMeter(context_window=4000))
    await compactor.compact(session, Summary())

    assert len(session.of_type(COMPACTION_START)) == len(session.of_type(COMPACTION_END)) == 1
    assert len(session.of_type(COMPACTION_SUMMARY)) == 1


async def test_a_failed_summarizer_closes_the_bracket_and_changes_nothing() -> None:
    """The surface must survive a failure intact -- a half-compacted session is
    worse than an uncompacted one."""
    session = conversation(20)
    surface_before = session.surface()
    compactor = Compactor(meter=TokenMeter(context_window=4000))

    result = await compactor.compact(session, Failing())

    assert not result.ok
    assert "exploded" in result.error
    assert session.surface() == surface_before
    assert len(session.of_type(COMPACTION_END)) == 1


async def test_an_empty_summary_is_refused() -> None:
    """Shadowing the conversation with nothing is worse than not compacting:
    the model loses the history and gains no account of it."""
    session = conversation(20)
    surface_before = session.surface()
    compactor = Compactor(meter=TokenMeter(context_window=4000))

    result = await compactor.compact(session, Empty())

    assert not result.ok
    assert session.surface() == surface_before


async def test_a_crashed_compaction_blocks_the_next_one() -> None:
    """An unmatched start means a summarizer died mid-transaction. Starting a
    second compaction on top of the first one's wreckage is not recovery."""
    session = conversation(20)
    session.append(COMPACTION_START, {"compaction": "crashed"})
    compactor = Compactor(meter=TokenMeter(context_window=4000))

    result = await compactor.compact(session, Summary())

    assert not result.ok
    assert "did not finish" in result.error


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def test_pruning_keeps_the_head_and_tail_of_an_oversized_result() -> None:
    """The head says what the tool returned and how it began; the tail says how
    it ended. The middle of a long listing is what nobody refers back to."""
    session = Session()
    session.append(USER_MESSAGE, {"text": "read it"})
    session.append(ASSISTANT_MESSAGE, {"text": "", "calls": ["read"]})
    session.append(
        TOOL_RESULT,
        {"name": "read", "call_id": "1", "ok": True, "content": "HEAD" + "x" * 9000 + "TAIL"},
    )

    result = Compactor().prune(session)

    assert result.ok
    surviving = [event for event in session.surface_events() if event.type == TOOL_RESULT][-1]
    content = surviving.data["content"]
    assert content.startswith("HEAD")
    assert content.endswith("TAIL")
    assert "removed by compaction" in content
    assert len(content) < 9000


def test_pruning_leaves_small_results_alone() -> None:
    session = Session()
    session.append(ASSISTANT_MESSAGE, {"text": "", "calls": ["read"]})
    session.append(TOOL_RESULT, {"name": "read", "content": "short"})

    result = Compactor().prune(session)

    assert not result.ok
    assert result.shadowed_seqs == ()


def test_pruning_costs_no_model_call() -> None:
    """Which is why it is tried first: it often relieves the pressure on its
    own, and only then is a summarizing request spent."""
    session = Session()
    session.append(ASSISTANT_MESSAGE, {"text": "", "calls": ["read"]})
    session.append(TOOL_RESULT, {"name": "read", "content": "x" * 9000})

    # No summarizer is passed at all: prune cannot need one.
    assert Compactor().prune(session).ok


# --------------------------------------------------------------------------
# wired into the loop
# --------------------------------------------------------------------------


class Verbose(Adapter):
    """Long answers, and a summarizer that answers when asked to summarize."""

    name = "verbose"

    def __init__(self) -> None:
        self.summaries = 0

    async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
        if "compacting the earlier part" in system:
            self.summaries += 1
            return ModelReply(text="EARLIER: a long discussion happened.")
        return ModelReply(text="Understood. " + "Here is a detailed explanation. " * 60)


async def test_a_long_conversation_stays_inside_the_window() -> None:
    """The point of the whole file. Without compaction this ends with the
    provider refusing the request."""
    adapter = Verbose()
    agent = HarnessAgent(
        adapter,
        ToolRegistry(Context(name="t")),
        system="be koe",
        compactor=Compactor(meter=TokenMeter(context_window=8_000)),
    )

    peak = 0.0
    for index in range(20):
        await agent.ask(f"question {index}")
        peak = max(peak, agent.compactor.meter.measure(agent.session).pressure)

    assert peak < 1.0, f"peaked at {peak:.0%} of the window"
    assert adapter.summaries >= 1, "compaction never ran"
    # The bracket closed every time.
    assert len(agent.session.of_type(COMPACTION_START)) == len(
        agent.session.of_type(COMPACTION_END)
    )


async def test_compaction_does_not_end_a_turn_when_it_fails() -> None:
    """It is what keeps a long conversation possible; it is not worth ending a
    turn over."""

    class BrokenCompactor(Compactor):
        def should_compact(self, measurement: Any) -> bool:
            raise RuntimeError("meter exploded")

    agent = HarnessAgent(
        Verbose(),
        ToolRegistry(Context(name="t")),
        compactor=BrokenCompactor(meter=TokenMeter(context_window=8_000)),
    )
    outcome = await agent.ask("go")
    assert outcome.reason == "completed"


def test_the_transcript_a_summarizer_reads_names_who_said_what() -> None:
    session = Session()
    session.append(USER_MESSAGE, {"text": "what is in the file?"})
    session.append(ASSISTANT_MESSAGE, {"text": "reading it", "calls": ["read"]})
    session.append(TOOL_RESULT, {"name": "read", "content": "contents"})

    rendered = render_transcript(session, session.surface())

    assert "User: what is in the file?" in rendered
    assert "Assistant: reading it" in rendered
    assert "Tool read returned: contents" in rendered
