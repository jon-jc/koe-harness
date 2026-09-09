"""Plugin discovery, activation, and the record of what is mounted.

The kernel already knows how to *run* a plugin. What was missing is everything
around that: finding one that was not compiled in, deciding whether it should
be running, and being able to turn it off without editing code. This module is
that layer, and it holds four positions.

**A plugin is a Python module with a well-known entry point.** Not a manifest
file beside a module, not a zip with metadata. The module declares itself with
module-level ``KOE_PLUGIN`` metadata and an ``apply`` (or a callable named in
the metadata), so the thing you read and the thing that runs are the same file.
A separate manifest is a second source of truth that goes stale.

**Discovery is explicit about where it looked.** A plugin that fails to import
is reported with its traceback rather than silently skipped, because the
failure mode of quiet skipping is a user who installed something, sees nothing,
and has no way to find out why.

**Disabling is unmounting, not a flag the plugin checks.** The kernel's scoped
teardown means a disabled plugin's tools, event handlers and services are gone,
not dormant. Any design where a plugin is asked to behave as if it were off
depends on every plugin author getting that right.

**Third-party code is trusted code.** A koe plugin runs in-process with full
Python privileges — this is a local developer tool, and the boundary that
matters is the one at install time. Saying so plainly beats implying a sandbox
that does not exist; the settings panel says it too, next to the button.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from koe.kernel.plugin import normalize

logger = logging.getLogger(__name__)

#: The attribute a plugin module sets to declare itself.
DECLARATION = "KOE_PLUGIN"

STATE_FILE = "plugins.json"


@dataclass(slots=True)
class PluginRecord:
    """One discovered plugin and what became of it."""

    name: str
    #: "builtin" or the path it was loaded from.
    origin: str
    description: str = ""
    version: str = ""
    #: Services it needs before it can run; the kernel gates activation on these.
    inject: tuple[str, ...] = ()
    builtin: bool = False
    enabled: bool = True
    active: bool = False
    #: Populated when import or activation failed. Non-empty means broken.
    error: str = ""
    _apply: Any = field(default=None, repr=False)
    _fork: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "description": self.description,
            "version": self.version,
            "inject": list(self.inject),
            "builtin": self.builtin,
            "enabled": self.enabled,
            "active": self.active,
            "error": self.error,
        }


class PluginManager:
    """Finds plugins, mounts the enabled ones, and can unmount them again."""

    def __init__(
        self, ctx: Any, *, directory: Path | None = None, state: Path | None = None
    ) -> None:
        self._ctx = ctx
        self._directory = directory
        self._state_path = state
        self._records: dict[str, PluginRecord] = {}
        self._disabled: set[str] = self._load_disabled()

    # -- state -------------------------------------------------------------

    def _load_disabled(self) -> set[str]:
        if self._state_path is None or not self._state_path.exists():
            return set()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return set(data.get("disabled", []))
        except (OSError, ValueError, AttributeError):
            # A corrupt state file must not stop the app from booting. The
            # cost of getting this wrong is that a plugin someone turned off
            # comes back on, which they can see and fix.
            logger.warning("plugin state unreadable; treating every plugin as enabled")
            return set()

    def _save_disabled(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"disabled": sorted(self._disabled)}, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("could not persist plugin state to %s", self._state_path)

    # -- registration ------------------------------------------------------

    def add_builtin(
        self, name: str, apply: Any, *, description: str = "", inject: tuple[str, ...] = ()
    ) -> None:
        """Register a plugin that ships with koe.

        Built-ins go through exactly the same record, activation and teardown
        path as a third-party plugin. If they had a shortcut, the shortcut
        would be the thing that worked and the plugin path would rot.
        """
        self._records[name] = PluginRecord(
            name=name,
            origin="builtin",
            description=description,
            inject=inject,
            builtin=True,
            enabled=name not in self._disabled,
            _apply=apply,
        )

    def discover(self) -> list[PluginRecord]:
        """Import every plugin module in the plugins directory."""
        if self._directory is None or not self._directory.is_dir():
            return []

        found: list[PluginRecord] = []
        for path in sorted(self._directory.iterdir()):
            module_path = self._module_file(path)
            if module_path is None:
                continue
            record = self._import(path.stem, module_path)
            self._records[record.name] = record
            found.append(record)
        return found

    @staticmethod
    def _module_file(path: Path) -> Path | None:
        """The importable file for a directory entry, or None if it is not one."""
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
            return path
        if path.is_dir() and (path / "__init__.py").is_file():
            return path / "__init__.py"
        return None

    def _import(self, name: str, module_path: Path) -> PluginRecord:
        origin = str(module_path)
        try:
            spec = importlib.util.spec_from_file_location(f"koe_plugin_{name}", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {module_path}")
            module = importlib.util.module_from_spec(spec)
            # Registered before exec so a plugin split across modules can
            # import itself, which is the normal shape once one grows.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except Exception:
            logger.exception("plugin %s failed to import", name)
            return PluginRecord(
                name=name,
                origin=origin,
                enabled=False,
                error=traceback.format_exc(limit=6),
            )

        declared = getattr(module, DECLARATION, {}) or {}
        apply = getattr(module, str(declared.get("apply", "apply")), None) or getattr(
            module, "plugin", None
        )
        if apply is None:
            return PluginRecord(
                name=name,
                origin=origin,
                enabled=False,
                error=(
                    f"{module_path.name} defines no apply(ctx, config). Add one, or name "
                    f"another callable with {DECLARATION} = {{'apply': '<name>'}}."
                ),
            )

        plugin_name = str(declared.get("name", name))
        return PluginRecord(
            name=plugin_name,
            origin=origin,
            description=str(declared.get("description", "")),
            version=str(declared.get("version", "")),
            inject=tuple(declared.get("inject", ())),
            enabled=plugin_name not in self._disabled,
            _apply=apply,
        )

    # -- lifecycle ---------------------------------------------------------

    def activate_all(self) -> None:
        for record in self._records.values():
            if record.enabled and not record.active and not record.error:
                self._activate(record)

    def _activate(self, record: PluginRecord) -> None:
        try:
            spec = normalize(record._apply)
            # The name the operator sees wins over whatever the callable is
            # called, so a plugin cannot appear twice under two names.
            spec.name = record.name
            spec.inject = record.inject or spec.inject
            record._fork = self._ctx.plugin(spec)
            record.active = True
            record.error = ""
        except Exception:
            logger.exception("plugin %s failed to activate", record.name)
            record.active = False
            record.error = traceback.format_exc(limit=6)

    def _deactivate(self, record: PluginRecord) -> None:
        fork, record._fork = record._fork, None
        record.active = False
        if fork is None:
            return
        try:
            fork.dispose()
        except Exception:
            logger.exception("plugin %s failed to unload cleanly", record.name)
            record.error = traceback.format_exc(limit=6)

    def set_enabled(self, name: str, enabled: bool) -> PluginRecord:
        """Turn a plugin on or off, taking effect immediately."""
        record = self._records.get(name)
        if record is None:
            raise KeyError(name)

        record.enabled = enabled
        if enabled:
            self._disabled.discard(name)
            if not record.active and record._apply is not None:
                self._activate(record)
        else:
            self._disabled.add(name)
            self._deactivate(record)
        self._save_disabled()
        return record

    def reload(self) -> list[PluginRecord]:
        """Re-scan the directory, unmounting anything that disappeared."""
        for record in list(self._records.values()):
            if not record.builtin:
                self._deactivate(record)
                self._records.pop(record.name, None)
        self.discover()
        self.activate_all()
        return self.records()

    # -- inspection --------------------------------------------------------

    def records(self) -> list[PluginRecord]:
        return sorted(self._records.values(), key=lambda r: (not r.builtin, r.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugins": [record.to_dict() for record in self.records()],
            "directory": str(self._directory) if self._directory else "",
        }
