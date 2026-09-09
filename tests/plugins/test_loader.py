"""Plugin discovery and lifecycle.

The property worth defending is that **disabling a plugin unmounts it**. A
design where a disabled plugin is asked to behave as though it were off relies
on every plugin author implementing that correctly; this one relies on the
kernel's scoped teardown, so a disabled plugin's tools are gone rather than
dormant. That is what these tests check, through the tool registry, because
observing it from outside is the only way to know it actually happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koe.kernel import Context
from koe.plugins import PluginManager
from koe.tools import ToolRegistry, ToolRun, ToolSpec

SIMPLE = """
KOE_PLUGIN = {
    "name": "greeter",
    "description": "Adds a greeting tool.",
    "version": "1.0.0",
}


def apply(ctx, config=None):
    registry = ctx.get("tools")
    if registry is None:
        return

    async def greet(args, run):
        return "hello"

    from koe.tools import ToolSpec

    spec = ToolSpec(
        name="greet",
        description="Say hello.",
        parameters={"type": "object", "properties": {}},
        execute=greet,
        source="greeter",
    )
    ctx.scope.collect("tool:greet", registry.register(spec))
"""

BROKEN = "this is not valid python ((("

NO_ENTRY = """
KOE_PLUGIN = {"name": "inert"}
# deliberately defines no apply()
"""

RENAMED_ENTRY = """
KOE_PLUGIN = {"name": "renamed", "apply": "mount"}


def mount(ctx, config=None):
    ctx.provide("renamed_ran", True)
"""


@pytest.fixture
def harness(tmp_path: Path) -> tuple[Context, ToolRegistry, PluginManager, Path]:
    directory = tmp_path / "plugins"
    directory.mkdir()
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    ctx.provide("tools", registry)
    manager = PluginManager(ctx, directory=directory, state=tmp_path / "plugins.json")
    return ctx, registry, manager, directory


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_a_module_is_discovered_with_its_metadata(harness: Any) -> None:
    _, _, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")

    found = manager.discover()

    assert [r.name for r in found] == ["greeter"]
    assert found[0].description == "Adds a greeting tool."
    assert found[0].version == "1.0.0"
    assert not found[0].builtin


def test_a_package_directory_is_discovered(harness: Any) -> None:
    _, _, manager, directory = harness
    package = directory / "bundled"
    package.mkdir()
    (package / "__init__.py").write_text(SIMPLE, encoding="utf-8")

    assert [r.name for r in manager.discover()] == ["greeter"]


def test_private_modules_are_skipped(harness: Any) -> None:
    """`_helpers.py` beside a plugin is support code, not a second plugin."""
    _, _, manager, directory = harness
    (directory / "_helpers.py").write_text(SIMPLE, encoding="utf-8")

    assert manager.discover() == []


def test_a_broken_plugin_reports_its_error_rather_than_vanishing(harness: Any) -> None:
    """Silent skipping leaves a user with no way to find out why nothing happened."""
    _, _, manager, directory = harness
    (directory / "broken.py").write_text(BROKEN, encoding="utf-8")

    record = manager.discover()[0]

    assert record.error
    assert "SyntaxError" in record.error
    assert not record.enabled


def test_a_plugin_with_no_entry_point_says_what_to_add(harness: Any) -> None:
    _, _, manager, directory = harness
    (directory / "inert.py").write_text(NO_ENTRY, encoding="utf-8")

    record = manager.discover()[0]

    assert "apply(ctx, config)" in record.error


def test_the_entry_point_can_be_renamed(harness: Any) -> None:
    ctx, _, manager, directory = harness
    (directory / "renamed.py").write_text(RENAMED_ENTRY, encoding="utf-8")

    manager.discover()
    manager.activate_all()

    assert ctx.get("renamed_ran") is True


def test_one_broken_plugin_does_not_stop_the_others(harness: Any) -> None:
    _, registry, manager, directory = harness
    (directory / "broken.py").write_text(BROKEN, encoding="utf-8")
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")

    manager.discover()
    manager.activate_all()

    assert "greet" in registry


def test_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """A fresh install has no plugins directory, and that is the normal case."""
    manager = PluginManager(Context(name="test"), directory=tmp_path / "absent")
    assert manager.discover() == []


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_activation_contributes_the_plugin_s_tools(harness: Any) -> None:
    _, registry, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")

    manager.discover()
    manager.activate_all()

    assert "greet" in registry
    assert manager.records()[0].active


def test_disabling_unmounts_rather_than_hiding(harness: Any) -> None:
    """The property the whole design rests on."""
    _, registry, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")
    manager.discover()
    manager.activate_all()
    assert "greet" in registry

    manager.set_enabled("greeter", False)

    assert "greet" not in registry
    assert not manager.records()[0].active


def test_re_enabling_remounts(harness: Any) -> None:
    _, registry, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")
    manager.discover()
    manager.activate_all()
    manager.set_enabled("greeter", False)

    manager.set_enabled("greeter", True)

    assert "greet" in registry


def test_a_disabled_plugin_stays_disabled_across_restarts(harness: Any, tmp_path: Path) -> None:
    """Otherwise turning something off lasts until the next launch."""
    _, _, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")
    manager.discover()
    manager.activate_all()
    manager.set_enabled("greeter", False)

    fresh_ctx = Context(name="restart")
    fresh_registry = ToolRegistry(fresh_ctx)
    fresh_ctx.provide("tools", fresh_registry)
    restarted = PluginManager(fresh_ctx, directory=directory, state=tmp_path / "plugins.json")
    restarted.discover()
    restarted.activate_all()

    assert "greet" not in fresh_registry
    assert not restarted.records()[0].enabled


def test_enabling_an_unknown_plugin_raises(harness: Any) -> None:
    _, _, manager, _ = harness
    with pytest.raises(KeyError):
        manager.set_enabled("nonesuch", True)


def test_a_corrupt_state_file_does_not_stop_the_app(tmp_path: Path) -> None:
    """A user can re-disable a plugin; they cannot recover from a dead app."""
    state = tmp_path / "plugins.json"
    state.write_text("{not json", encoding="utf-8")

    manager = PluginManager(Context(name="test"), directory=tmp_path, state=state)

    assert manager.records() == []


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------


def test_a_builtin_goes_through_the_same_path(harness: Any) -> None:
    """If built-ins had a shortcut, the shortcut is what would stay working."""
    _, registry, manager, _ = harness

    def builtin(ctx: Any, config: Any = None) -> None:
        spec = ToolSpec(
            name="builtin_tool",
            description="From a built-in.",
            parameters={"type": "object", "properties": {}},
            execute=lambda args, run: "ok",
        )
        ctx.scope.collect("tool:builtin_tool", registry.register(spec))

    manager.add_builtin("core", builtin, description="Core tools.")
    manager.activate_all()
    assert "builtin_tool" in registry

    manager.set_enabled("core", False)
    assert "builtin_tool" not in registry


def test_builtins_are_listed_before_third_party(harness: Any) -> None:
    _, _, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")
    manager.add_builtin("core", lambda ctx, config=None: None)
    manager.discover()

    assert [r.name for r in manager.records()] == ["core", "greeter"]


@pytest.mark.asyncio
async def test_a_discovered_tool_actually_runs(harness: Any) -> None:
    """End to end: a file on disk becomes a callable tool."""
    _, registry, manager, directory = harness
    (directory / "greeter.py").write_text(SIMPLE, encoding="utf-8")
    manager.discover()
    manager.activate_all()

    result = await registry.call("greet")

    assert result.ok
    assert result.content == "hello"


def test_reload_drops_a_plugin_that_was_deleted(harness: Any) -> None:
    _, registry, manager, directory = harness
    module = directory / "greeter.py"
    module.write_text(SIMPLE, encoding="utf-8")
    manager.discover()
    manager.activate_all()
    assert "greet" in registry

    module.unlink()
    manager.reload()

    assert "greet" not in registry
    assert manager.records() == []


def test_reload_keeps_builtins(harness: Any) -> None:
    _, _, manager, _ = harness
    manager.add_builtin("core", lambda ctx, config=None: None)

    manager.reload()

    assert [r.name for r in manager.records()] == ["core"]


def test_an_unused_run_argument_is_still_passed(harness: Any) -> None:
    """Pins the tool body signature the plugin docs promise."""
    run = ToolRun(tool="t", call_id="c")
    assert run.owner == "default"
    assert not run.cancelled.is_set()
