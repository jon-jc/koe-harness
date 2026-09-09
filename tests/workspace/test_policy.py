"""Read-before-edit.

The tests are written against the *tools*, through the registry, rather than
against the policy's internals. That is deliberate: the policy's entire claim
is that it works without the tools knowing it exists, and the only way to check
that claim is to exercise the path a model actually takes.

Two properties matter most, and each fails silently without a test. An
unguarded overwrite destroys a file with nothing in the transcript saying so.
And a guard that cannot be removed is a fork rather than a policy — so the
last section runs the same tools with the plugin unmounted and asserts they
still work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from koe.kernel import Context
from koe.tools import ToolRegistry
from koe.tools.builtin import workspace_tools, workspace_write_tools
from koe.workspace import Workspace, read_before_edit

pytestmark = pytest.mark.asyncio


@pytest.fixture
def guarded(tmp_path: Path) -> tuple[ToolRegistry, Path]:
    """Read tools, write tools, and the policy — the shipped composition."""
    (tmp_path / "notes.md").write_text("original\n", encoding="utf-8")
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    ctx.provide("tools", registry)
    ctx.provide("workspace", Workspace(tmp_path))

    ctx.plugin(workspace_tools)
    ctx.plugin(read_before_edit)
    ctx.plugin(workspace_write_tools)
    return registry, tmp_path


@pytest.fixture
def unguarded(tmp_path: Path) -> tuple[ToolRegistry, Path]:
    """The same tools with the policy left out."""
    (tmp_path / "notes.md").write_text("original\n", encoding="utf-8")
    ctx = Context(name="test")
    registry = ToolRegistry(ctx)
    ctx.provide("tools", registry)
    ctx.provide("workspace", Workspace(tmp_path))

    ctx.plugin(workspace_tools)
    ctx.plugin(workspace_write_tools)
    return registry, tmp_path


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


async def test_overwriting_an_unread_file_is_refused(guarded: Any) -> None:
    """The failure this exists to prevent: a silent, total loss."""
    registry, root = guarded

    result = await registry.call("write_file", {"path": "notes.md", "content": "clobbered"})

    assert not result.ok
    assert "without reading it first" in result.detail
    assert (root / "notes.md").read_text(encoding="utf-8") == "original\n"


async def test_editing_an_unread_file_is_refused(guarded: Any) -> None:
    registry, _ = guarded

    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "original", "new_text": "changed"}
    )

    assert not result.ok
    assert "without reading it first" in result.detail


async def test_reading_first_authorizes_the_write(guarded: Any) -> None:
    registry, root = guarded

    await registry.call("read_file", {"path": "notes.md"})
    result = await registry.call("write_file", {"path": "notes.md", "content": "replaced\n"})

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "replaced\n"


async def test_reading_first_authorizes_the_edit(guarded: Any) -> None:
    registry, root = guarded

    await registry.call("read_file", {"path": "notes.md"})
    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "original", "new_text": "edited"}
    )

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "edited\n"


async def test_a_new_file_needs_no_read(guarded: Any) -> None:
    """Otherwise a model can never create a file: the first write always fails."""
    registry, root = guarded

    result = await registry.call("write_file", {"path": "fresh.md", "content": "new\n"})

    assert result.ok
    assert (root / "fresh.md").read_text(encoding="utf-8") == "new\n"


async def test_creating_inside_a_new_directory_works(guarded: Any) -> None:
    registry, root = guarded

    result = await registry.call("write_file", {"path": "docs/deep/note.md", "content": "hi"})

    assert result.ok
    assert (root / "docs" / "deep" / "note.md").is_file()


# --------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------


async def test_a_file_that_changed_since_the_read_is_refused(guarded: Any) -> None:
    """Overwriting anyway would discard whatever the other writer did."""
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"})

    (root / "notes.md").write_text("someone else got here first\n" * 4, encoding="utf-8")

    result = await registry.call("write_file", {"path": "notes.md", "content": "mine"})

    assert not result.ok
    assert "changed since you read it" in result.detail
    assert "someone else" in (root / "notes.md").read_text(encoding="utf-8")


async def test_re_reading_clears_the_staleness(guarded: Any) -> None:
    """The refusal names a remedy; the remedy has to actually work."""
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"})
    (root / "notes.md").write_text("theirs\n" * 4, encoding="utf-8")

    await registry.call("read_file", {"path": "notes.md"})
    result = await registry.call("write_file", {"path": "notes.md", "content": "mine\n"})

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "mine\n"


async def test_a_write_leaves_the_caller_holding_a_current_view(guarded: Any) -> None:
    """Re-reading after every write is ceremony that teaches models to ignore the rule."""
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"})
    await registry.call("write_file", {"path": "notes.md", "content": "first\n"})

    result = await registry.call("write_file", {"path": "notes.md", "content": "second\n"})

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "second\n"


# --------------------------------------------------------------------------
# absence is an observation
# --------------------------------------------------------------------------


async def test_reading_a_missing_file_then_creating_it(guarded: Any) -> None:
    """ "I checked and it was not there" is a state, not an absence of one."""
    registry, root = guarded

    missing = await registry.call("read_file", {"path": "absent.md"})
    assert not missing.ok

    result = await registry.call("write_file", {"path": "absent.md", "content": "created\n"})

    assert result.ok
    assert (root / "absent.md").read_text(encoding="utf-8") == "created\n"


async def test_a_write_cannot_clobber_a_file_created_since_the_check(guarded: Any) -> None:
    """The race the guarded-create flow exists to lose safely."""
    registry, root = guarded
    await registry.call("read_file", {"path": "racy.md"})

    (root / "racy.md").write_text("someone else created it\n", encoding="utf-8")

    result = await registry.call("write_file", {"path": "racy.md", "content": "mine"})

    assert not result.ok
    assert "changed since you read it" in result.detail
    assert "someone else" in (root / "racy.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# ownership
# --------------------------------------------------------------------------


async def test_one_owner_s_read_does_not_authorize_another_s_write(guarded: Any) -> None:
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"}, owner="alice")

    result = await registry.call(
        "write_file", {"path": "notes.md", "content": "mallory"}, owner="mallory"
    )

    assert not result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "original\n"


# --------------------------------------------------------------------------
# the edit contract
# --------------------------------------------------------------------------


async def test_an_ambiguous_edit_is_refused_rather_than_applied_to_the_first(
    guarded: Any,
) -> None:
    """Editing the first of several matches is how a tool silently does the wrong thing."""
    registry, root = guarded
    (root / "notes.md").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    await registry.call("read_file", {"path": "notes.md"})

    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "x = 1", "new_text": "x = 9"}
    )

    assert not result.ok
    assert "appears 2 times" in result.detail
    assert (root / "notes.md").read_text(encoding="utf-8") == "x = 1\ny = 2\nx = 1\n"


async def test_enough_context_makes_an_ambiguous_edit_work(guarded: Any) -> None:
    registry, root = guarded
    (root / "notes.md").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    await registry.call("read_file", {"path": "notes.md"})

    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "y = 2\nx = 1", "new_text": "y = 2\nx = 9"}
    )

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "x = 1\ny = 2\nx = 9\n"


async def test_an_edit_that_matches_nothing_says_so(guarded: Any) -> None:
    registry, _ = guarded
    await registry.call("read_file", {"path": "notes.md"})

    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "absent", "new_text": "x"}
    )

    assert not result.ok
    assert "does not appear" in result.detail


async def test_editing_a_file_observed_absent_is_refused(guarded: Any) -> None:
    registry, _ = guarded
    await registry.call("read_file", {"path": "nothere.md"})

    result = await registry.call(
        "edit_file", {"path": "nothere.md", "old_text": "a", "new_text": "b"}
    )

    assert not result.ok
    assert "nothing to edit" in result.detail


# --------------------------------------------------------------------------
# containment still applies
# --------------------------------------------------------------------------


async def test_a_write_cannot_escape_the_workspace(guarded: Any) -> None:
    registry, _ = guarded

    result = await registry.call("write_file", {"path": "../escaped.md", "content": "nope"})

    assert not result.ok
    assert "outside the workspace" in result.detail


async def test_an_absolute_write_path_is_refused(guarded: Any) -> None:
    registry, _ = guarded

    result = await registry.call("write_file", {"path": "/etc/passwd", "content": "nope"})

    assert not result.ok


# --------------------------------------------------------------------------
# removing the guard
# --------------------------------------------------------------------------


async def test_without_the_policy_the_tools_still_work(unguarded: Any) -> None:
    """A guard that cannot be removed is a fork, not a policy.

    This is the property that lets one tool set serve a deployment that wants
    the guard and one that does not — and it only holds because the tools
    never learned the policy exists.
    """
    registry, root = unguarded

    result = await registry.call("write_file", {"path": "notes.md", "content": "unguarded\n"})

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "unguarded\n"


async def test_without_the_policy_an_edit_needs_no_read(unguarded: Any) -> None:
    registry, root = unguarded

    result = await registry.call(
        "edit_file", {"path": "notes.md", "old_text": "original", "new_text": "changed"}
    )

    assert result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "changed\n"


# --------------------------------------------------------------------------
# atomicity
# --------------------------------------------------------------------------


async def test_a_write_leaves_no_temporary_file_behind(guarded: Any) -> None:
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"})
    await registry.call("write_file", {"path": "notes.md", "content": "done\n"})

    assert [p.name for p in root.iterdir() if ".tmp" in p.name] == []


async def test_a_failed_write_leaves_the_original_intact(guarded: Any, tmp_path: Path) -> None:
    """Writing in place leaves a truncated file and no way to tell it happened."""
    registry, root = guarded
    await registry.call("read_file", {"path": "notes.md"})

    # A directory where the file should be: the rename cannot succeed.
    (root / "blocked.md").mkdir()
    result = await registry.call("write_file", {"path": "blocked.md", "content": "x"})

    assert not result.ok
    assert (root / "notes.md").read_text(encoding="utf-8") == "original\n"
