"""Workspace file access, fenced to one root.

The code viewer and the model-facing file tools are the same capability seen
from two sides, so they share one service rather than each reaching for
``pathlib`` and each getting the containment check subtly wrong.

**Containment is resolved, not textual.** ``..`` is the obvious attack and the
easy one to catch; the ones that actually get through a naive check are a
symlink pointing out of the tree, a Windows 8.3 short name, and a drive-letter
absolute path smuggled in where a relative one was expected. Every path is
resolved to a real location first and then tested for ancestry against the
resolved root, because only after resolution are those the same question.

**Reads are bounded, and say when they were cut.** A tool that returns a 400 MB
log does not answer a question, it fills a context window. Both the byte cap
and the line window are reported back, so a caller that needs the rest knows
there is a rest.

**Binary files are refused, not mangled.** Decoding a PNG as UTF-8 with
``errors="replace"`` produces thousands of replacement characters that look
like data. A null byte in the first block is the cheap, reliable signal, and
saying "this is binary" is more useful than any lossy rendering of it.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Read cap per file. Generous for source, far below anything that would
#: threaten a context window.
MAX_READ_BYTES = 2_000_000

#: Directories never worth walking. Skipped at the directory level rather than
#: filtered per file, so a node_modules with 30k files costs one check.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".idea",
        ".vscode",
        "target",
    }
)

#: Extension → the token a syntax highlighter wants. Kept here rather than in
#: the client so a plugin adding a language does not need a UI change.
LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".css": "css",
    ".html": "html",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".sql": "sql",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".txt": "text",
    ".cfg": "ini",
    ".ini": "ini",
    ".iss": "ini",
    ".spec": "python",
}


class WorkspaceError(Exception):
    """A refused or impossible file operation, with a routable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _dominant_newline(text: str) -> str:
    """The line ending a file mostly uses.

    Whichever is more common wins, so one stray ending in a mixed file does
    not flip the whole convention.
    """
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _is_absolute(text: str) -> bool:
    """Whether a client-supplied path is absolute, on any platform.

    Deliberately not ``Path(text).is_absolute()``: that answers for the host
    it runs on, and a server can receive a Windows path while running on
    Linux. Every form is refused everywhere, so the same request is refused
    identically wherever koe is deployed.
    """
    if text.startswith("/"):  # POSIX absolute, and UNC once slashes are folded
        return True
    # A drive letter: "C:", "c:/x". Two characters is enough to decide.
    return len(text) >= 2 and text[1] == ":" and text[0].isalpha()


@dataclass(frozen=True, slots=True)
class Entry:
    """One directory entry."""

    name: str
    path: str
    is_dir: bool
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "is_dir": self.is_dir, "size": self.size}


@dataclass(frozen=True, slots=True)
class FileView:
    """A bounded read of one file."""

    path: str
    text: str
    language: str
    lines: int
    #: Total lines in the file, which may exceed `lines` when a window was asked for.
    total_lines: int
    size: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "text": self.text,
            "language": self.language,
            "lines": self.lines,
            "total_lines": self.total_lines,
            "size": self.size,
            "truncated": self.truncated,
        }


class Workspace:
    """Read access to one directory tree, and nothing outside it."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root or Path.cwd()).resolve()

    @property
    def root(self) -> Path:
        return self._root

    # -- containment -------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Resolve a workspace-relative path, or refuse it.

        Two checks, in this order, and the order is the whole design.

        **An absolute path is refused outright**, before anything else touches
        it. Quietly reinterpreting ``/etc/passwd`` as a path *relative to the
        workspace* is contained — it lands under the root and cannot escape —
        and it is still wrong: the caller asked for one file and would be
        handed a different one, or a confusing "not found" for a file that
        plainly exists. This shipped broken and only Linux caught it: on
        Windows a drive-absolute path replaces the root during joining and
        fails the containment check by accident, so the same bug passed there.

        **Everything else is checked after resolution.** Before resolution,
        ``notes/../../etc/passwd`` and ``notes/link-to-etc`` look completely
        different; afterwards they are the same problem, and one comparison
        catches both — along with the 8.3 short names and case folding that a
        textual check on Windows misses.
        """
        text = (relative or "").strip().replace("\\", "/")
        if not text or text == ".":
            return self._root
        if _is_absolute(text):
            raise WorkspaceError(
                "outside_workspace", f"{relative!r} is absolute; paths must be workspace-relative"
            )

        candidate = (self._root / text).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise WorkspaceError("outside_workspace", f"{relative!r} is outside the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        """A path as the client names it: workspace-relative, forward slashes."""
        try:
            return path.resolve().relative_to(self._root).as_posix()
        except ValueError:
            return path.name

    # -- reading -----------------------------------------------------------

    def list_dir(self, relative: str = "") -> list[Entry]:
        """One directory, directories first then files, both alphabetical."""
        target = self.resolve(relative)
        if not target.is_dir():
            raise WorkspaceError("not_a_directory", f"{relative!r} is not a directory")

        entries: list[Entry] = []
        for child in target.iterdir():
            if child.name.startswith(".") and child.name not in {".github", ".claude"}:
                continue
            if child.is_dir() and child.name in IGNORED_DIRS:
                continue
            try:
                size = child.stat().st_size if child.is_file() else 0
            except OSError:
                # A broken symlink or a file that vanished mid-walk should not
                # take down a directory listing.
                continue
            entries.append(
                Entry(
                    name=child.name,
                    path=self.relative(child),
                    is_dir=child.is_dir(),
                    size=size,
                )
            )
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return entries

    def read(self, relative: str, *, start: int = 1, count: int | None = None) -> FileView:
        """Read a file, or a line window of one. Lines are 1-based."""
        target = self.resolve(relative)
        if not target.is_file():
            raise WorkspaceError("not_found", f"{relative!r} is not a file")

        size = target.stat().st_size
        raw = target.read_bytes()[:MAX_READ_BYTES]
        truncated = size > MAX_READ_BYTES

        if b"\x00" in raw[:8192]:
            raise WorkspaceError("binary", f"{relative!r} is a binary file")

        # Normalized to LF before anyone sees it. A model writes LF and a
        # Windows-authored file holds CRLF, so matching without this makes
        # `edit_file` fail on exactly the files a Windows user has — while
        # looking, in every log, as though the text simply was not there.
        # The original ending is restored on write, so the file keeps its
        # convention and nothing in the repository churns.
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        all_lines = text.splitlines()
        total = len(all_lines)

        if count is not None:
            begin = max(1, start) - 1
            window = all_lines[begin : begin + max(1, count)]
            text = "\n".join(window)
            shown = len(window)
            truncated = truncated or shown < total
        else:
            shown = total

        return FileView(
            path=self.relative(target),
            text=text,
            language=LANGUAGES.get(target.suffix.lower(), "text"),
            lines=shown,
            total_lines=total,
            size=size,
            truncated=truncated,
        )

    # -- searching ---------------------------------------------------------

    def glob(self, pattern: str, *, limit: int = 500) -> list[str]:
        """Paths matching a glob, newest first.

        Newest first because the question behind a glob is almost always
        "what did I just touch", and an alphabetical answer buries it.
        """
        matches: list[tuple[float, str]] = []
        for path in self._walk():
            relative = self.relative(path)
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
                try:
                    matches.append((path.stat().st_mtime, relative))
                except OSError:
                    continue
            if len(matches) >= limit * 4:
                break
        matches.sort(reverse=True)
        return [relative for _, relative in matches[:limit]]

    def grep(
        self,
        pattern: str,
        *,
        glob: str | None = None,
        limit: int = 200,
        ignore_case: bool = True,
    ) -> list[dict[str, Any]]:
        """Regex search across the tree, with the matching line."""
        try:
            expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise WorkspaceError("bad_pattern", f"invalid regular expression: {exc}") from exc

        hits: list[dict[str, Any]] = []
        for path in self._walk():
            relative = self.relative(path)
            if glob and not (fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(path.name, glob)):
                continue
            try:
                raw = path.read_bytes()[:MAX_READ_BYTES]
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            for number, line in enumerate(raw.decode("utf-8", "replace").splitlines(), start=1):
                if expression.search(line):
                    hits.append(
                        {
                            "path": relative,
                            "line": number,
                            # Long minified lines are the usual reason a grep
                            # result is unreadable.
                            "text": line[:400],
                        }
                    )
                    if len(hits) >= limit:
                        return hits
        return hits

    def _walk(self) -> list[Path]:
        """Every file worth looking at, ignored directories pruned."""
        found: list[Path] = []
        for base, dirs, files in os.walk(self._root):
            # Mutating `dirs` in place is what makes os.walk skip the subtree
            # rather than walk it and discard the results.
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
            for name in files:
                if not name.startswith("."):
                    found.append(Path(base) / name)
        return found

    # -- writing -----------------------------------------------------------

    def version(self, relative: str) -> str:
        """A cheap fingerprint of a file's current state.

        Size and mtime rather than a hash of the contents: the question this
        answers is "has this changed since I read it", and stat is orders of
        magnitude cheaper than hashing a file on every guarded write. It can
        miss a change that preserves both — a same-length edit inside one
        filesystem timestamp tick — which is a real gap and an acceptable one
        for a guard whose purpose is catching *concurrent editors*, not
        defeating a deliberate forgery.

        An absent file has a version too, and it is the empty string. That is
        what lets "I checked, it was not there" be a state the guard can
        enforce rather than an absence of information.
        """
        try:
            target = self.resolve(relative)
        except WorkspaceError:
            raise
        if not target.is_file():
            return ""
        stat = target.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def write(self, relative: str, text: str, *, expect: str | None = None) -> FileView:
        """Replace a file's contents, atomically.

        `expect` is the version the caller last saw. When supplied it is
        checked immediately before the replace and the write is refused if it
        no longer matches, which is what makes "read, then write" mean
        something under a concurrent editor.

        The write itself goes to a temporary file in the same directory and is
        then renamed over the target. A partial write that crashed halfway
        leaves the original intact; writing in place leaves a truncated file
        and no way to tell it happened.
        """
        target = self.resolve(relative)
        if target.is_dir():
            raise WorkspaceError("not_a_file", f"{relative!r} is a directory")

        if expect is not None:
            current = self.version(relative)
            if current != expect:
                raise WorkspaceError(
                    "stale",
                    f"{relative!r} changed since it was read — read it again, then retry",
                )

        # Restore whatever the file already used. Rewriting a CRLF file with
        # LF endings turns a one-line change into a diff of the whole file.
        if target.is_file():
            try:
                existing = target.read_bytes()[:MAX_READ_BYTES].decode("utf-8", "replace")
                newline = _dominant_newline(existing)
            except OSError:
                newline = "\n"
            if newline != "\n":
                text = text.replace("\r\n", "\n").replace("\n", newline)

        target.parent.mkdir(parents=True, exist_ok=True)
        # Same directory, because os.replace is only atomic within a
        # filesystem and a temp directory may be on another one.
        temporary = target.with_name(f".{target.name}.koe-{os.getpid()}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8", newline="")
            # `replace` rather than `rename`: rename refuses to overwrite on
            # Windows, which is precisely the case this is for.
            temporary.replace(target)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise WorkspaceError("write_failed", f"could not write {relative!r}: {exc}") from exc

        return self.read(relative)

    def edit(
        self, relative: str, old: str, new: str, *, expect: str | None = None
    ) -> tuple[FileView, int]:
        """Replace one exact occurrence of `old` with `new`.

        Exact string replacement rather than line numbers, and it refuses when
        `old` appears more than once. Both are the same decision: a line number
        is stale the moment anything above it changes, and an ambiguous match
        edited at the first occurrence is the failure mode where a tool
        silently changes the wrong thing. Making the caller supply enough
        surrounding text to be unique turns that into a refusal it can fix.
        """
        if not old:
            raise WorkspaceError("empty_match", "the text to replace must not be empty")

        view = self.read(relative)
        occurrences = view.text.count(old)
        if occurrences == 0:
            raise WorkspaceError("no_match", f"the text to replace does not appear in {relative!r}")
        if occurrences > 1:
            raise WorkspaceError(
                "ambiguous",
                f"the text to replace appears {occurrences} times in {relative!r} — "
                "include enough surrounding lines to make it unique",
            )

        updated = self.write(relative, view.text.replace(old, new, 1), expect=expect)
        return updated, occurrences

    # -- summary -----------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {"root": str(self._root), "name": self._root.name}
