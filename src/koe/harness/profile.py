"""Composing a harness declaratively, instead of in code.

Ported from DeepSeek Harness's `boot` and `bundle` groups (MIT). dsh's central
claim is that everything is a plugin; a profile is what makes that claim usable,
because a set of plugins you can only change by editing source is a build, not a
composition.

koe mounts its plugins in Python: `add_builtin("terminal", ...)` and so on. That
is fine until someone wants a koe without the terminal, or with compaction tuned
for a smaller context window, or with a plugin they wrote — and then it is a
fork.

**A profile is an ordered list of rows.** Each has an `id`, the plugin `name` it
mounts, optional `config`, and `disabled`. Row order carries no load semantics:
activation is driven by service availability, exactly as in koe's kernel, so a
plugin that needs `tools` waits for `tools` regardless of where it sits.

**Patches address rows by id, and the last write wins per row.** This is what
lets layers compose: koe ships a base profile, a deployment adds a patch, the
user adds another, and none of them has to know what the others contain. Only
the rows a layer names are affected.

**A patch replaces a row's whole `config` rather than merging into it.** dsh is
explicit about this and it is the right choice, though it looks unhelpful at
first. Merging means a user who sets one field inherits every other field from
a layer they cannot see, so the effective config is the result of a computation
nobody has written down. Replacing means the row's config is exactly what its
last writer said.

**TOML rather than dsh's YAML.** `tomllib` is in the standard library from 3.11
and koe already configures itself in TOML, so a profile costs no dependency and
uses the syntax the project's readers already know. The structure is dsh's.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProfileError(Exception):
    """A profile could not be read or resolved."""


@dataclass(frozen=True, slots=True)
class PluginRow:
    """One plugin in a profile."""

    #: Stable identity a patch addresses. Distinct from `name` on purpose: two
    #: rows may mount the same plugin with different configuration, and a patch
    #: has to be able to name one of them.
    id: str
    #: The registered plugin this row mounts. Defaults to the id, because they
    #: are usually the same and repeating it is noise.
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ProfileError("a profile row needs an id")
        if not self.name:
            object.__setattr__(self, "name", self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": dict(self.config),
            "disabled": self.disabled,
        }


@dataclass(frozen=True, slots=True)
class Profile:
    """An ordered set of plugin rows."""

    rows: tuple[PluginRow, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Any:
        return iter(self.rows)

    def get(self, row_id: str) -> PluginRow | None:
        return next((row for row in self.rows if row.id == row_id), None)

    @property
    def enabled(self) -> tuple[PluginRow, ...]:
        return tuple(row for row in self.rows if not row.disabled)

    def patch(self, operations: list[dict[str, Any]]) -> Profile:
        """Apply one layer of operations, returning a new profile.

        Three operations, matching dsh's: `insert` adds rows (or replaces one
        that already has the id, so a layer can restate a row without knowing
        whether an earlier layer created it), `update` changes fields of an
        existing row, and `remove` drops one.

        Returns a new profile rather than mutating: a layer that half-applied
        and then failed would leave a composition nobody wrote.
        """
        rows = list(self.rows)
        index = {row.id: position for position, row in enumerate(rows)}

        for operation in operations:
            kind = str(operation.get("op") or "insert")

            if kind == "remove":
                target = str(operation.get("id", ""))
                if target in index:
                    rows.pop(index[target])
                    index = {row.id: position for position, row in enumerate(rows)}
                continue

            if kind == "update":
                target = str(operation.get("id", ""))
                position = index.get(target)
                if position is None:
                    raise ProfileError(f"patch updates unknown row {target!r}")
                current = rows[position]
                rows[position] = replace(
                    current,
                    name=str(operation.get("name") or current.name),
                    # Replaced, not merged. See the module docstring.
                    config=dict(operation["config"]) if "config" in operation else current.config,
                    disabled=bool(operation.get("disabled", current.disabled)),
                )
                continue

            if kind != "insert":
                raise ProfileError(f"unknown patch operation {kind!r}")

            row = PluginRow(
                id=str(operation.get("id", "")),
                name=str(operation.get("name", "")),
                config=dict(operation.get("config") or {}),
                disabled=bool(operation.get("disabled", False)),
            )
            position = index.get(row.id)
            if position is None:
                index[row.id] = len(rows)
                rows.append(row)
            else:
                # Restating a row is how a later layer takes ownership of it
                # without having to know whether an earlier one created it.
                rows[position] = row

        return Profile(rows=tuple(rows))

    def to_dict(self) -> dict[str, Any]:
        return {"plugins": [row.to_dict() for row in self.rows]}


def parse(text: str) -> Profile:
    """Read a profile from TOML.

    ```toml
    [[plugins]]
    id = "terminal"
    disabled = true

    [[plugins]]
    id = "user-vocabulary"
    [plugins.config]
    path = "/etc/koe/vocabulary.txt"
    ```
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"profile is not valid TOML: {exc}") from exc

    raw = document.get("plugins")
    if raw is None:
        return Profile()
    if not isinstance(raw, list):
        raise ProfileError("`plugins` must be a list of tables")

    rows: list[PluginRow] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ProfileError("each plugin row must be a table")
        row = PluginRow(
            id=str(entry.get("id", "")),
            name=str(entry.get("name", "")),
            config=dict(entry.get("config") or {}),
            disabled=bool(entry.get("disabled", False)),
        )
        if row.id in seen:
            # Two rows with one id makes every patch that names it ambiguous,
            # which is a composition whose meaning depends on iteration order.
            raise ProfileError(f"duplicate plugin id {row.id!r}")
        seen.add(row.id)
        rows.append(row)
    return Profile(rows=tuple(rows))


def parse_patch(text: str) -> list[dict[str, Any]]:
    """Read a patch layer from TOML.

    ```toml
    [[patch]]
    op = "update"
    id = "terminal"
    disabled = true
    ```
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"patch is not valid TOML: {exc}") from exc

    raw = document.get("patch")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ProfileError("`patch` must be a list of tables")
    return [dict(entry) for entry in raw if isinstance(entry, dict)]


def load(path: Path, *, patches: list[Path] | None = None) -> Profile:
    """Read a profile and apply patch layers over it, in order.

    A missing profile is an empty one rather than an error: koe runs with its
    built-in composition and a profile only ever narrows or extends it, so
    "no file" and "a file that changes nothing" should behave the same.
    """
    profile = Profile()
    if path.is_file():
        profile = parse(path.read_text(encoding="utf-8"))

    for patch_path in patches or []:
        if not patch_path.is_file():
            continue
        profile = profile.patch(parse_patch(patch_path.read_text(encoding="utf-8")))
    return profile


def apply_profile(manager: Any, profile: Profile) -> list[str]:
    """Set a plugin manager's state from a profile. Returns what it changed.

    Rows naming plugins the build does not have are reported rather than
    raising: a profile written for a koe with an extra plugin should still
    start the koe you have, minus that plugin, and say so. Failing instead
    would make one stale row a startup failure.
    """
    changes: list[str] = []
    for row in profile.rows:
        record = manager.record(row.name)
        if record is None:
            changes.append(f"{row.id}: no plugin named {row.name!r} in this build")
            continue

        if row.config:
            record.config = dict(row.config)
            changes.append(f"{row.name}: configured")
        if row.disabled == record.enabled:
            manager.set_enabled(row.name, not row.disabled)
            changes.append(f"{row.name}: {'disabled' if row.disabled else 'enabled'}")
    return changes
