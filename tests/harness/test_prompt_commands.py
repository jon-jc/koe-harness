"""Prompt assembly, and the commands a user types at the harness."""

from __future__ import annotations

from typing import Any

import pytest

from koe.agent.loop import Adapter, ModelReply
from koe.harness import HarnessAgent
from koe.harness.commands import Command, CommandRegistry, CommandResult, builtin_commands
from koe.harness.compaction import Compactor
from koe.harness.prompt import ORDERS, PromptAssemblyError, SystemPrompt
from koe.harness.tokens import TokenMeter
from koe.kernel.context import Context
from koe.tools.registry import ToolRegistry


class Echo(Adapter):
    name = "echo"

    def __init__(self) -> None:
        self.systems: list[str] = []

    async def reply(self, messages: Any, *, system: str, tools: Any) -> ModelReply:
        self.systems.append(system)
        return ModelReply(text="ok")


def agent(**kwargs: Any) -> HarnessAgent:
    return HarnessAgent(
        kwargs.pop("adapter", Echo()),
        kwargs.pop("tools", ToolRegistry(Context(name="t"))),
        compactor=Compactor(meter=TokenMeter(context_window=4_000)),
        **kwargs,
    )


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------


def test_unloading_a_plugin_takes_its_instructions_with_it() -> None:
    """The failure this exists to prevent: a prompt that tells the model it can
    run shell commands after the terminal plugin was turned off."""
    prompt = SystemPrompt()
    prompt.section("identity", "You are koe.", order="IDENTITY")
    dispose = prompt.section("terminal", "You can run shell commands.", order="TOOL_TERMINAL")

    assert "shell commands" in prompt.render()
    dispose()
    assert "shell commands" not in prompt.render()
    assert "You are koe." in prompt.render()


def test_sections_assemble_in_slot_order_not_registration_order() -> None:
    prompt = SystemPrompt()
    prompt.section("last", "LAST", order=9_000)
    prompt.section("first", "FIRST", order="IDENTITY")
    assert prompt.render().index("FIRST") < prompt.render().index("LAST")


def test_a_tie_breaks_on_name_so_the_bytes_are_stable() -> None:
    """A prompt whose bytes depend on plugin load order invalidates a
    provider's prefix cache on a run that changed nothing."""
    first = SystemPrompt()
    first.section("bbb", "B", order=100)
    first.section("aaa", "A", order=100)

    second = SystemPrompt()
    second.section("aaa", "A", order=100)
    second.section("bbb", "B", order=100)

    assert first.render() == second.render()


def test_a_dynamic_section_describes_the_current_state() -> None:
    """Rather than the state at registration, which is what a string would."""
    count = 0

    def text() -> str:
        return f"{count} terminals are open."

    prompt = SystemPrompt()
    prompt.section("terminals", text, order="TOOL_TERMINAL")

    assert "0 terminals" in prompt.render()
    count = 3
    assert "3 terminals" in prompt.render()


def test_a_section_that_raises_is_dropped_rather_than_taking_the_prompt() -> None:
    """Its absence is a worse prompt; its exception is no prompt at all."""

    def broken() -> str:
        raise RuntimeError("cannot describe myself")

    prompt = SystemPrompt()
    prompt.section("ok", "This part works.", order="IDENTITY")
    prompt.section("broken", broken, order="POLICY")

    assert prompt.render() == "This part works."


def test_a_duplicate_section_name_is_refused() -> None:
    prompt = SystemPrompt()
    prompt.section("identity", "one", order="IDENTITY")
    with pytest.raises(ValueError, match="already registered"):
        prompt.section("identity", "two", order="IDENTITY")


def test_a_stale_disposer_cannot_remove_a_later_section() -> None:
    prompt = SystemPrompt()
    dispose = prompt.section("slot", "original", order="POLICY")
    dispose()
    prompt.section("slot", "replacement", order="POLICY")
    dispose()
    assert "replacement" in prompt.render()


def test_an_unresolved_variable_refuses_rather_than_leaking() -> None:
    """A prompt containing a literal `{{workspace}}` is not degraded, it is an
    instruction the model will try to make sense of."""
    prompt = SystemPrompt()
    prompt.section("w", "Workspace is {{workspace}}.", order="WORKSPACE")

    with pytest.raises(PromptAssemblyError, match="workspace"):
        prompt.render()

    prompt.variable("workspace", "/srv/koe")
    assert prompt.render() == "Workspace is /srv/koe."


def test_two_sections_cannot_both_be_the_complete_prompt() -> None:
    """Picking one silently would make the prompt depend on load order."""
    prompt = SystemPrompt()
    prompt.section("a", "A", order="IDENTITY", complete=True)
    prompt.section("b", "B", order="POLICY", complete=True)
    with pytest.raises(PromptAssemblyError, match="complete prompt"):
        prompt.render()


def test_a_complete_section_replaces_everything_else() -> None:
    prompt = SystemPrompt()
    prompt.section("identity", "You are koe.", order="IDENTITY")
    prompt.section("override", "Ignore all of that.", order="POLICY", complete=True)
    assert prompt.render() == "Ignore all of that."


def test_the_slots_are_ordered_the_way_a_reader_would_want() -> None:
    assert ORDERS["IDENTITY"] < ORDERS["POLICY"] < ORDERS["TOOL_WORKSPACE"]
    assert ORDERS["TOOL_WORKSPACE"] < ORDERS["PERSONA"]


async def test_the_agent_uses_the_assembled_prompt() -> None:
    adapter = Echo()
    prompt = SystemPrompt()
    prompt.section("identity", "You are koe.", order="IDENTITY")
    await agent(adapter=adapter, prompt=prompt).ask("hello")
    assert adapter.systems[-1] == "You are koe."


async def test_a_failed_assembly_falls_back_rather_than_ending_the_turn() -> None:
    """A stale prompt still answers the question; a dead turn answers
    nothing."""
    adapter = Echo()
    prompt = SystemPrompt()
    prompt.section("broken", "needs {{missing}}", order="IDENTITY")

    outcome = await agent(adapter=adapter, prompt=prompt, system="fallback").ask("hello")

    assert outcome.reason == "completed"
    assert adapter.systems[-1] == "fallback"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", ("help", "")),
        ("/compact ", ("compact", "")),
        ("/compact everything", ("compact", "everything")),
        ("  /clear  ", ("clear", "")),
        ("hello", None),
        ("", None),
        ("/", None),
        # A path is not a command, and treating it as one would send a user's
        # message to a handler because it happened to start with a slash.
        ("/usr/bin/env is a path", None),
        ("/ leading space", None),
    ],
)
def test_only_a_leading_slash_and_a_name_is_a_command(
    text: str, expected: tuple[str, str] | None
) -> None:
    assert CommandRegistry.parse(text) == expected


async def test_a_command_runs_instead_of_a_turn() -> None:
    """It never reaches the inbox, so the model cannot read `/compact` as a
    request to talk about compacting."""
    adapter = Echo()
    subject = agent(adapter=adapter)

    result = await subject.command("/help")

    assert result is not None and result.ok
    assert adapter.systems == [], "a command spent a model call"
    assert not subject.inbox.pending


async def test_an_ordinary_message_is_not_a_command() -> None:
    assert await agent().command("what is in the file?") is None


async def test_an_unknown_command_lists_the_known_ones() -> None:
    result = await agent().command("/nope")
    assert result is not None and not result.ok
    assert "/compact" in result.text


async def test_a_command_that_takes_no_arguments_says_so() -> None:
    """The common failure is typing `/compact everything` and wondering which
    part was ignored."""
    result = await agent().command("/compact everything")
    assert result is not None and not result.ok
    assert "no arguments" in result.text


async def test_compacting_is_refused_while_a_turn_is_running() -> None:
    """A compaction racing a turn would replace the history that turn is
    deriving its next request from."""
    subject = agent()
    subject.followup("go")
    subject._phase = "running"

    result = await subject.command("/compact")

    assert result is not None and not result.ok
    assert "working" in result.text


async def test_clear_starts_a_new_session() -> None:
    subject = agent()
    await subject.ask("hello")
    before = subject.session.id

    result = await subject.command("/clear")

    assert result is not None and result.reload
    assert subject.session.id != before
    assert subject.session.derive_messages() == []


async def test_context_reports_where_the_window_went() -> None:
    """ "Why is it forgetting things" and "why is this costing so much" have the
    same answer, and it is otherwise invisible."""
    subject = agent()
    await subject.ask("hello")

    result = await subject.command("/context")

    assert result is not None and result.ok
    assert "tokens" in result.text
    assert "visible to the model" in result.text


async def test_a_failing_command_says_the_conversation_may_have_changed() -> None:
    """ "It did not work" is useless to someone deciding whether their
    conversation is intact."""

    async def explode(agent_: Any, rest: str) -> CommandResult:
        raise RuntimeError("boom")

    subject = agent()
    subject.commands.register(Command(name="boom", summary="fails", run=explode))

    result = await subject.command("/boom")

    assert result is not None and not result.ok
    assert "did not finish" in result.text
    assert "Check the session" in result.text


def test_the_builtins_are_all_registered() -> None:
    names = {entry["name"] for entry in builtin_commands().describe()}
    assert {"compact", "clear", "stop", "context", "help"} <= names


def test_a_duplicate_command_is_refused() -> None:
    registry = builtin_commands()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Command(name="help", summary="again", run=_noop))


async def _noop(agent_: Any, rest: str) -> CommandResult:
    return CommandResult(text="")
