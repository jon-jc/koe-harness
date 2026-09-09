"""Read-before-edit.

A model that writes a file it has not read is editing its idea of the file
rather than the file. Sometimes those agree. When they do not, the write is
silent and total: the previous contents are gone, and nothing in the
transcript says so. This plugin makes that impossible by refusing the write.

Four positions, three of them dsh's.

**It is a plugin, not a service the tools inject.** The tools call
``ctx.get("workspace")`` and know nothing about this module. Removing the
plugin leaves the bare provider's unconditional behaviour — mutation without a
guard — rather than an import error or a tool that cannot start. That is what
makes the guard a deployment choice instead of a fork.

**Absence is an observation.** Reading a file that is not there records
"confirmed absent", which authorizes creating it. Without that, a model can
never create a file at all: the first write always fails for want of a read it
cannot perform. With it, "I checked and it was not there" and "I never looked"
stay distinct, and only the second is refused.

**A stale read is refused, not merged.** The observation carries the version
the file had when it was read, and the write is refused if it no longer
matches. Overwriting anyway would discard whatever the other writer did; the
refusal names the remedy, which is to read again and retry.

**Observations are per-owner and not persisted.** They live for the process,
keyed by the agent that made them, so one agent's read does not authorize
another's write, and a restart means re-reading. Persisting them would mean
trusting a record of a file's state from before an unknown amount of drift —
which is exactly the thing the guard exists to catch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from koe.tools.registry import Denial, ToolError, ToolRun, ToolSpec
from koe.workspace.service import Workspace, WorkspaceError

logger = logging.getLogger(__name__)

#: Tools whose first argument names a file they are about to change.
GUARDED = frozenset({"write_file", "edit_file"})

#: Tools whose success means the caller has seen the file.
OBSERVING = frozenset({"read_file"})


@dataclass(frozen=True, slots=True)
class Observation:
    """What one owner last saw at one path.

    An empty `version` means the file was confirmed absent — a real state,
    distinct from never having looked, which is the absence of a record.
    """

    path: str
    version: str

    @property
    def absent(self) -> bool:
        return self.version == ""


class ObservationLog:
    """What each owner has read, and what it looked like at the time."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], Observation] = {}

    def record(self, owner: str, path: str, version: str) -> None:
        self._seen[(owner, path)] = Observation(path=path, version=version)

    def get(self, owner: str, path: str) -> Observation | None:
        return self._seen.get((owner, path))

    def forget(self, owner: str, path: str) -> None:
        self._seen.pop((owner, path), None)

    def __len__(self) -> int:
        return len(self._seen)


def read_before_edit(ctx: Any, config: Any = None) -> None:
    """Refuse a guarded write to a file the caller has not read."""
    workspace: Workspace | None = ctx.get("workspace")
    if workspace is None:
        return

    log = ObservationLog()
    ctx.provide("observations", log, replace=True)

    async def gate(tool: ToolSpec, run: ToolRun) -> Denial | None:
        if tool.name not in GUARDED:
            return None

        path = str(run.arguments.get("path") or "").strip()
        if not path:
            # Not this plugin's job to validate arguments; the tool will
            # reject it with a message aimed at the caller.
            return None

        try:
            current = workspace.version(path)
        except WorkspaceError:
            # An unresolvable path is the tool's refusal to make, and it will
            # explain containment better than a policy denial would.
            return None

        seen = log.get(run.owner, path)
        if seen is None:
            verb = "edit" if tool.name == "edit_file" else "overwrite"
            if current == "" and tool.name == "write_file":
                # Creating a file nobody has looked at is the one unguarded
                # case, and it is still worth a record: the write below
                # observes it, so a second write is guarded like any other.
                return None
            return Denial(
                f'cannot {verb} "{path}" without reading it first — '
                f"call read_file on it, then retry",
                ToolError.DENIED,
            )

        if seen.absent and tool.name == "edit_file":
            return Denial(
                f'"{path}" was not there when you read it, so there is nothing to edit',
                ToolError.DENIED,
            )

        if seen.version != current:
            return Denial(
                f'"{path}" changed since you read it — read it again, then retry',
                ToolError.DENIED,
            )
        return None

    async def observe(tool: ToolSpec, run: ToolRun, result: Any) -> None:
        """Record what a read saw, and what a write left behind."""
        path = str(run.arguments.get("path") or "").strip()
        if not path:
            return

        if tool.name in OBSERVING:
            # A *failed* read is recorded too, and only when the reason is
            # that the file is not there. That is the whole "absence is an
            # observation" rule: without it a model can never create a file,
            # because reading a missing one fails and leaves no record, so the
            # write that follows is refused for want of a read that could not
            # have succeeded. A read that failed for any other reason —
            # binary, outside the workspace — records nothing.
            try:
                version = workspace.version(path)
            except WorkspaceError:
                return
            if result.ok or version == "":
                log.record(run.owner, path, version)
        elif tool.name in GUARDED and result.ok:
            # A successful write leaves the caller holding a current view, so
            # a follow-up edit does not need another read. Re-reading after
            # every write is ceremony that teaches a model to ignore the rule.
            try:
                log.record(run.owner, path, workspace.version(path))
            except WorkspaceError:
                log.forget(run.owner, path)

    ctx.scope.collect("policy:pre", ctx.on("tools/pre-execute", gate))
    ctx.scope.collect("policy:post", ctx.on("tools/post-execute", observe))
    logger.info("read-before-edit policy mounted")


def observed_absent(ctx: Any, owner: str, path: str) -> bool:
    """Whether `owner` has confirmed `path` does not exist.

    Exposed for a tool that wants to phrase its own message differently; the
    enforcement is entirely in the gate above.
    """
    log = ctx.get("observations")
    if log is None:
        return False
    seen = log.get(owner, path)
    return seen is not None and seen.absent
