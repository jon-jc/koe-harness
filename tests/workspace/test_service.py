"""Workspace access, and the fence around it.

The containment tests are the point of this file. Everything else here is
convenience; ``resolve`` refusing to leave the root is the property that
decides whether exposing a file service over a local HTTP server was a
reasonable thing to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from koe.workspace import Workspace, WorkspaceError


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\nprint('bye')\n", encoding="utf-8")
    (tmp_path / "src" / "util.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# 議事録\n\nkoe\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    return tmp_path


# --------------------------------------------------------------------------
# containment
# --------------------------------------------------------------------------


def test_a_traversal_is_refused(tree: Path) -> None:
    workspace = Workspace(tree)
    with pytest.raises(WorkspaceError) as caught:
        workspace.resolve("../../etc/passwd")
    assert caught.value.code == "outside_workspace"


def test_an_absolute_path_is_refused(tree: Path) -> None:
    """A client that sends an absolute path is not asking for the workspace."""
    outside = "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/passwd"
    with pytest.raises(WorkspaceError):
        Workspace(tree).resolve(outside)


def test_a_traversal_that_returns_inside_is_allowed(tree: Path) -> None:
    """`src/../README.md` is a real path inside the root, not an escape."""
    workspace = Workspace(tree)
    assert workspace.resolve("src/../README.md") == (tree / "README.md").resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privilege on Windows")
def test_a_symlink_out_of_the_tree_is_refused(
    tree: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The case a textual '..' check misses entirely."""
    elsewhere = tmp_path_factory.mktemp("outside")
    (elsewhere / "secret.txt").write_text("nope", encoding="utf-8")
    (tree / "escape").symlink_to(elsewhere)

    with pytest.raises(WorkspaceError) as caught:
        Workspace(tree).resolve("escape/secret.txt")
    assert caught.value.code == "outside_workspace"


def test_backslashes_are_normalized(tree: Path) -> None:
    """Windows clients send them, and they must not defeat the check."""
    workspace = Workspace(tree)
    assert workspace.resolve("src\\main.py") == (tree / "src" / "main.py").resolve()
    with pytest.raises(WorkspaceError):
        workspace.resolve("..\\..\\secret")


def test_the_root_itself_resolves(tree: Path) -> None:
    workspace = Workspace(tree)
    assert workspace.resolve("") == tree.resolve()
    assert workspace.resolve(".") == tree.resolve()


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------


def test_listing_hides_noise(tree: Path) -> None:
    names = {entry.name for entry in Workspace(tree).list_dir()}
    assert "src" in names
    assert "README.md" in names
    assert "node_modules" not in names
    assert ".git" not in names


def test_directories_come_first(tree: Path) -> None:
    entries = Workspace(tree).list_dir()
    assert entries[0].is_dir
    assert not entries[-1].is_dir


def test_listing_a_file_is_an_error(tree: Path) -> None:
    with pytest.raises(WorkspaceError) as caught:
        Workspace(tree).list_dir("README.md")
    assert caught.value.code == "not_a_directory"


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_reading_a_file(tree: Path) -> None:
    view = Workspace(tree).read("src/main.py")
    assert view.language == "python"
    assert view.total_lines == 2
    assert "print('hi')" in view.text


def test_japanese_survives_the_round_trip(tree: Path) -> None:
    view = Workspace(tree).read("README.md")
    assert "議事録" in view.text
    assert view.language == "markdown"


def test_a_line_window(tree: Path) -> None:
    view = Workspace(tree).read("src/main.py", start=2, count=1)
    assert view.text == "print('bye')"
    assert view.lines == 1
    assert view.total_lines == 2
    assert view.truncated


def test_a_binary_file_is_refused_rather_than_mangled(tree: Path) -> None:
    """Decoding a PNG produces thousands of replacement characters that look like data."""
    with pytest.raises(WorkspaceError) as caught:
        Workspace(tree).read("logo.png")
    assert caught.value.code == "binary"


def test_a_missing_file_is_an_error(tree: Path) -> None:
    with pytest.raises(WorkspaceError) as caught:
        Workspace(tree).read("nope.py")
    assert caught.value.code == "not_found"


# --------------------------------------------------------------------------
# searching
# --------------------------------------------------------------------------


def test_glob_matches_by_name_and_path(tree: Path) -> None:
    workspace = Workspace(tree)
    assert "src/main.py" in workspace.glob("*.py")
    assert "src/util.ts" in workspace.glob("src/*.ts")


def test_glob_does_not_walk_ignored_directories(tree: Path) -> None:
    assert Workspace(tree).glob("*.js") == []


def test_grep_finds_a_line_with_its_number(tree: Path) -> None:
    hits = Workspace(tree).grep(r"print\('bye'\)")
    assert len(hits) == 1
    assert hits[0]["path"] == "src/main.py"
    assert hits[0]["line"] == 2


def test_grep_can_be_restricted_by_glob(tree: Path) -> None:
    assert Workspace(tree).grep("print", glob="*.ts") == []


def test_grep_skips_binary_files(tree: Path) -> None:
    """Otherwise every search matches noise inside images."""
    assert Workspace(tree).grep("PNG") == []


def test_an_invalid_regex_is_reported_not_raised_raw(tree: Path) -> None:
    with pytest.raises(WorkspaceError) as caught:
        Workspace(tree).grep("(unclosed")
    assert caught.value.code == "bad_pattern"
