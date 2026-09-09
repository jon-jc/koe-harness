"""The tool registry and its pipeline.

Two properties here are security-relevant rather than merely correct, and both
are pinned as tests because both fail silently: host-only fields must never
reach a model request, and a tool that raises must not take the turn with it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from koe.kernel import Context
from koe.tools import (
    Denial,
    ToolError,
    ToolInvocationError,
    ToolRegistry,
    ToolRun,
    ToolSpec,
)


def spec(name: str = "echo", **overrides: Any) -> ToolSpec:
    async def echo(args: dict[str, Any], run: ToolRun) -> Any:
        return args.get("text", "")

    fields: dict[str, Any] = {
        "name": name,
        "description": "Echo the text back.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        "execute": echo,
    }
    fields.update(overrides)
    return ToolSpec(**fields)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_a_registered_tool_is_callable_and_listed() -> None:
    registry = ToolRegistry()
    registry.register(spec())
    assert "echo" in registry
    assert [s["name"] for s in registry.schemas()] == ["echo"]


def test_the_disposer_removes_the_tool() -> None:
    """A registration is an effect: the plugin that made it can unmake it."""
    registry = ToolRegistry()
    dispose = registry.register(spec())
    dispose()
    assert "echo" not in registry
    assert len(registry) == 0


def test_disposing_twice_is_harmless() -> None:
    """Scopes tear down more than once; a disposer must tolerate it."""
    registry = ToolRegistry()
    dispose = registry.register(spec())
    dispose()
    dispose()
    assert len(registry) == 0


def test_a_disposer_does_not_remove_a_replacement() -> None:
    """A stale disposer must not delete the tool that took the name."""
    registry = ToolRegistry()
    dispose = registry.register(spec())
    dispose()
    registry.register(spec(description="the replacement"))
    dispose()
    assert "echo" in registry


def test_a_duplicate_name_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec())


def test_tools_are_listed_in_a_stable_order() -> None:
    """The model's tool list should not reshuffle between requests."""
    registry = ToolRegistry()
    for name in ("zebra", "alpha", "middle"):
        registry.register(spec(name))
    assert [s["name"] for s in registry.schemas()] == ["alpha", "middle", "zebra"]


# --------------------------------------------------------------------------
# the allowlist
# --------------------------------------------------------------------------


def test_host_only_fields_never_reach_the_model() -> None:
    """The whole reason `schemas()` is an allowlist rather than a filter."""
    registry = ToolRegistry()
    registry.register(spec(timeout_s=5.0, dangerous=True, source="third-party"))

    projected = registry.schemas()[0]

    assert set(projected) == {"name", "description", "parameters"}
    assert "timeout_s" not in projected
    assert "dangerous" not in projected
    assert "source" not in projected


def test_an_operator_still_sees_the_host_fields() -> None:
    """They are hidden from the model, not from the settings panel."""
    registry = ToolRegistry()
    registry.register(spec(dangerous=True, source="third-party"))

    described = registry.describe()[0]

    assert described["dangerous"] is True
    assert described["source"] == "third-party"


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_call_returns_content() -> None:
    registry = ToolRegistry()
    registry.register(spec())
    result = await registry.call("echo", {"text": "こんにちは"})

    assert result.ok
    assert result.content == "こんにちは"
    assert result.error is None


@pytest.mark.asyncio
async def test_an_unknown_tool_names_the_alternatives() -> None:
    """The usual cause is a model inventing a tool; a list is what corrects it."""
    registry = ToolRegistry()
    registry.register(spec("read_file"))

    result = await registry.call("reed_file")

    assert not result.ok
    assert result.error is ToolError.NO_TOOL
    assert "read_file" in result.detail


@pytest.mark.asyncio
async def test_a_crashing_tool_becomes_a_result_not_an_exception() -> None:
    """Raising through the agent loop would discard the whole turn."""

    async def explode(args: dict[str, Any], run: ToolRun) -> Any:
        raise RuntimeError("the disk is on fire")

    registry = ToolRegistry()
    registry.register(spec(execute=explode))

    result = await registry.call("echo")

    assert not result.ok
    assert result.error is ToolError.FAILED
    assert "the disk is on fire" in result.detail


@pytest.mark.asyncio
async def test_a_failure_is_legible_in_the_content_the_model_reads() -> None:
    """A model only reads `content`; an error code beside it is invisible."""
    registry = ToolRegistry()
    result = await registry.call("nonexistent")

    assert result.content
    assert "no_tool" in result.content


@pytest.mark.asyncio
async def test_bad_arguments_are_distinguished_from_a_crash() -> None:
    """The model can fix one of these and not the other."""

    async def picky(args: dict[str, Any], run: ToolRun) -> Any:
        raise ToolInvocationError("path is required")

    registry = ToolRegistry()
    registry.register(spec(execute=picky))

    result = await registry.call("echo")

    assert result.error is ToolError.BAD_ARGUMENTS


@pytest.mark.asyncio
async def test_a_synchronous_tool_body_works() -> None:
    """Not every tool needs to be async; only the await is conditional."""

    def plain(args: dict[str, Any], run: ToolRun) -> Any:
        return "sync"

    registry = ToolRegistry()
    registry.register(spec(execute=plain))

    assert (await registry.call("echo")).content == "sync"


@pytest.mark.asyncio
async def test_a_slow_tool_times_out() -> None:
    async def slow(args: dict[str, Any], run: ToolRun) -> Any:
        await asyncio.sleep(5)

    registry = ToolRegistry()
    registry.register(spec(execute=slow, timeout_s=0.05))

    result = await registry.call("echo")

    assert result.error is ToolError.TIMEOUT


@pytest.mark.asyncio
async def test_a_large_result_is_truncated_and_says_so() -> None:
    """A silently truncated result is one the model treats as complete."""

    async def firehose(args: dict[str, Any], run: ToolRun) -> Any:
        return "x" * 200_000

    registry = ToolRegistry()
    registry.register(spec(execute=firehose))

    result = await registry.call("echo")

    assert result.truncated
    assert "truncated" in result.content
    assert len(result.content) < 200_000


@pytest.mark.asyncio
async def test_a_structured_value_survives_beside_its_rendering() -> None:
    """The model gets text; a UI that can render more should not have to parse it."""

    async def structured(args: dict[str, Any], run: ToolRun) -> Any:
        return {"files": ["a.py", "b.py"]}

    registry = ToolRegistry()
    registry.register(spec(execute=structured))

    result = await registry.call("echo")

    assert result.value == {"files": ["a.py", "b.py"]}
    assert "a.py" in result.content


# --------------------------------------------------------------------------
# the policy seam
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_policy_can_deny_a_call() -> None:
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    registry.register(spec(dangerous=True))

    async def gate(tool: ToolSpec, run: ToolRun) -> Any:
        return Denial("dangerous tools need approval") if tool.dangerous else None

    ctx.on("tools/pre-execute", gate)

    result = await registry.call("echo", {"text": "hi"})

    assert not result.ok
    assert result.error is ToolError.DENIED
    assert "approval" in result.detail


@pytest.mark.asyncio
async def test_without_a_policy_the_tool_simply_runs() -> None:
    """Removing the guard degrades to unguarded, not to broken."""
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    registry.register(spec(dangerous=True))

    result = await registry.call("echo", {"text": "hi"})

    assert result.ok


@pytest.mark.asyncio
async def test_post_execute_fires_for_a_failure_too() -> None:
    """A listener that only sees successes cannot audit anything.

    This shipped wrong: failures returned early and never reached the event,
    so the read-before-edit policy could not record that a read failed because
    the file was absent — which is the observation that authorizes creating it.
    """
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)

    async def explode(args: dict[str, Any], run: ToolRun) -> Any:
        raise RuntimeError("boom")

    registry.register(spec(execute=explode))
    seen: list[Any] = []

    async def audit(tool: ToolSpec, run: ToolRun, result: Any) -> None:
        seen.append((tool.name, result.ok, result.error))

    ctx.on("tools/post-execute", audit)
    await registry.call("echo")

    assert seen == [("echo", False, ToolError.FAILED)]


@pytest.mark.asyncio
async def test_post_execute_fires_for_a_denial() -> None:
    """A refused call is exactly what an audit log most wants to see."""
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    registry.register(spec(dangerous=True))
    seen: list[Any] = []

    async def gate(tool: ToolSpec, run: ToolRun) -> Any:
        return Denial("no") if tool.dangerous else None

    async def audit(tool: ToolSpec, run: ToolRun, result: Any) -> None:
        seen.append(result.error)

    ctx.on("tools/pre-execute", gate)
    ctx.on("tools/post-execute", audit)
    await registry.call("echo")

    assert seen == [ToolError.DENIED]


@pytest.mark.asyncio
async def test_an_unknown_tool_does_not_fire_post_execute() -> None:
    """There is no tool to hand the listener, so there is nothing to report."""
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    seen: list[Any] = []

    async def audit(tool: ToolSpec, run: ToolRun, result: Any) -> None:
        seen.append(tool)

    ctx.on("tools/post-execute", audit)
    await registry.call("nonesuch")

    assert seen == []


@pytest.mark.asyncio
async def test_post_execute_observes_the_result() -> None:
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    registry.register(spec())
    seen: list[Any] = []

    async def audit(tool: ToolSpec, run: ToolRun, result: Any) -> None:
        seen.append((tool.name, result.ok))

    ctx.on("tools/post-execute", audit)
    await registry.call("echo", {"text": "hi"})

    assert seen == [("echo", True)]
