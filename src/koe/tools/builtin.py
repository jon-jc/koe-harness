"""The tools that ship in the box, as plugins.

Nothing here is privileged. Each group is a plugin that registers on
``ctx.tools`` and returns disposers, so a deployment that does not want the
model reading files simply does not mount ``workspace_tools`` — there is no
flag to find and no core to patch. That is the property the whole plugin
argument rests on, and it is only true if the first-party tools obey it too.

The descriptions are written for a model, not for a human reading source. A
tool description is a prompt: it is the only thing standing between "the model
uses this correctly" and "the model guesses". Saying *when not to* use a tool
turns out to matter more than saying what it does.
"""

from __future__ import annotations

from typing import Any

from koe.tools.registry import ToolInvocationError, ToolRun, ToolSpec
from koe.workspace.service import Workspace, WorkspaceError


def _require(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInvocationError(f"{key!r} is required and must be a non-empty string")
    return value.strip()


def workspace_tools(ctx: Any, config: Any = None) -> None:
    """Read-only file access: list, read, glob, grep.

    Read-only is deliberate for the first cut. A write tool needs the
    read-before-edit policy and an approval path to be safe, and shipping the
    mutation half before the guard is how a harness earns a bad reputation in
    one release.
    """
    registry = ctx.get("tools")
    workspace: Workspace = ctx.get("workspace")
    if registry is None or workspace is None:
        return

    async def list_files(args: dict[str, Any], run: ToolRun) -> Any:
        path = str(args.get("path", "") or "")
        try:
            entries = workspace.list_dir(path)
        except WorkspaceError as exc:
            raise ToolInvocationError(str(exc)) from exc
        if not entries:
            return f"{path or '.'} is empty"
        return "\n".join(
            f"{'dir ' if e.is_dir else 'file'}  {e.path}" + ("" if e.is_dir else f"  ({e.size:,}B)")
            for e in entries
        )

    async def read_file(args: dict[str, Any], run: ToolRun) -> Any:
        path = _require(args, "path")
        start = int(args.get("start_line") or 1)
        count = args.get("line_count")
        try:
            view = workspace.read(path, start=start, count=int(count) if count else None)
        except WorkspaceError as exc:
            raise ToolInvocationError(str(exc)) from exc
        # Line numbers, because the next thing anyone does with a file read is
        # refer to a line in it.
        offset = start if count else 1
        body = "\n".join(
            f"{number:>6}  {line}"
            for number, line in enumerate(view.text.splitlines(), start=offset)
        )
        header = f"{view.path} ({view.total_lines:,} lines)"
        if view.truncated:
            header += " [truncated]"
        return f"{header}\n{body}"

    async def glob_files(args: dict[str, Any], run: ToolRun) -> Any:
        pattern = _require(args, "pattern")
        matches = workspace.glob(pattern)
        return "\n".join(matches) if matches else f"no files match {pattern!r}"

    async def grep_files(args: dict[str, Any], run: ToolRun) -> Any:
        pattern = _require(args, "pattern")
        glob = args.get("glob")
        try:
            hits = workspace.grep(pattern, glob=str(glob) if glob else None)
        except WorkspaceError as exc:
            raise ToolInvocationError(str(exc)) from exc
        if not hits:
            return f"no matches for {pattern!r}"
        return "\n".join(f"{h['path']}:{h['line']}: {h['text']}" for h in hits)

    specs = [
        ToolSpec(
            name="list_files",
            description=(
                "List the files and directories at a path in the workspace. "
                "Use it to orient yourself before reading. Build directories, "
                "caches and version-control internals are already excluded, so "
                "what you get back is the source."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory. Omit for the root.",
                    }
                },
            },
            execute=list_files,
            source="workspace",
        ),
        ToolSpec(
            name="read_file",
            description=(
                "Read a text file from the workspace, with line numbers. Pass "
                "start_line and line_count for a window rather than reading a "
                "large file whole. Binary files are refused rather than "
                "returned as noise."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "start_line": {"type": "integer", "description": "1-based first line."},
                    "line_count": {"type": "integer", "description": "How many lines to return."},
                },
                "required": ["path"],
            },
            execute=read_file,
            source="workspace",
        ),
        ToolSpec(
            name="glob_files",
            description=(
                "Find files by name pattern, most recently modified first. Use "
                "this when you know roughly what a file is called; use "
                "grep_files when you know what is inside it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob such as '**/*.py' or 'test_*.py'.",
                    }
                },
                "required": ["pattern"],
            },
            execute=glob_files,
            source="workspace",
        ),
        ToolSpec(
            name="grep_files",
            description=(
                "Search file contents with a regular expression and return the "
                "matching lines with their paths. Prefer this over reading "
                "whole files to look for something."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "glob": {
                        "type": "string",
                        "description": "Restrict to files matching this glob.",
                    },
                },
                "required": ["pattern"],
            },
            execute=grep_files,
            timeout_s=20.0,
            source="workspace",
        ),
    ]

    for spec in specs:
        # The disposer goes on the scope, so unloading this plugin removes
        # every tool it added and leaves none pointing at dead code.
        ctx.scope.collect(f"tool:{spec.name}", registry.register(spec))


def meeting_tools(ctx: Any, config: Any = None) -> None:
    """Tools over koe's own domain, so the assistant can answer about a meeting.

    This is the pairing that makes the harness more than a chat window bolted
    onto a transcriber: the same agent that can read the repository can also
    read the transcript that was just recorded, and the 議事録 built from it.
    """
    registry = ctx.get("tools")
    if registry is None:
        return

    async def current_transcript(args: dict[str, Any], run: ToolRun) -> Any:
        session = ctx.get("last_transcript")
        if not session:
            return "No meeting has been transcribed in this session yet."
        return session

    async def current_minutes(args: dict[str, Any], run: ToolRun) -> Any:
        minutes = ctx.get("last_minutes")
        if not minutes:
            return "No 議事録 has been generated in this session yet."
        return minutes

    specs = [
        ToolSpec(
            name="current_transcript",
            description=(
                "The transcript of the meeting recorded in this session, with "
                "speaker labels and timestamps. Empty until a recording has "
                "finished."
            ),
            parameters={"type": "object", "properties": {}},
            execute=current_transcript,
            source="meeting",
        ),
        ToolSpec(
            name="current_minutes",
            description=(
                "The generated 議事録 for this session: summary, decisions, and "
                "action items, each with the transcript quote it was verified "
                "against. Empty until minutes have been generated."
            ),
            parameters={"type": "object", "properties": {}},
            execute=current_minutes,
            source="meeting",
        ),
    ]
    for spec in specs:
        ctx.scope.collect(f"tool:{spec.name}", registry.register(spec))
