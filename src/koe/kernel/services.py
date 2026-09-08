"""The reactive service registry.

Services are the harness's dependency-injection surface: an ASR backend, an
LLM client, the cost ledger, the transcript store. Two things make this
registry more than a dict.

**Ownership.** A service is registered *by a scope*. When that scope is
disposed the service unregisters itself, so a plugin cannot leave a dangling
client behind after it is unloaded.

**Reactivity.** Anything can subscribe to a service name and be told when it
appears, is replaced, or goes away. That is what powers dependency-gated
plugin activation: the registry does not know what a plugin is, it just
announces changes, and the context turns those announcements into
start/stop decisions.

This is what makes a provider swap safe at runtime. Replacing the `asr`
service tears down exactly the plugins that depend on ASR, in dependency
order, and rebuilds them against the new backend -- while the websocket
handling live audio, which depends on none of it, keeps running.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from koe.kernel.errors import ServiceConflict, ServiceNotFound

logger = logging.getLogger(__name__)


class ServiceChange(StrEnum):
    ADDED = "added"
    REPLACED = "replaced"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class ServiceEvent:
    name: str
    change: ServiceChange
    old: Any = None
    new: Any = None


Observer = Callable[[ServiceEvent], None]

_MISSING = object()


class ServiceRegistry:
    """Name -> service, with ownership and change notification."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._owners: dict[str, str] = {}
        self._observers: dict[str, list[Observer]] = {}
        self._global_observers: list[Observer] = []

    # -- reads ---------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._values

    def get(self, name: str, default: Any = _MISSING) -> Any:
        if name in self._values:
            return self._values[name]
        if default is not _MISSING:
            return default
        raise ServiceNotFound(name, list(self._values))

    def names(self) -> list[str]:
        return sorted(self._values)

    def owner(self, name: str) -> str | None:
        return self._owners.get(name)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)

    # -- writes --------------------------------------------------------------

    def set(self, name: str, value: Any, *, owner: str = "root", replace: bool = False) -> None:
        existed = name in self._values
        if existed and not replace:
            raise ServiceConflict(name)
        old = self._values.get(name)
        self._values[name] = value
        self._owners[name] = owner
        logger.debug("service %r %s by %s", name, "replaced" if existed else "registered", owner)
        self._notify(
            ServiceEvent(
                name=name,
                change=ServiceChange.REPLACED if existed else ServiceChange.ADDED,
                old=old,
                new=value,
            )
        )

    def delete(self, name: str) -> None:
        if name not in self._values:
            return
        old = self._values.pop(name)
        self._owners.pop(name, None)
        logger.debug("service %r removed", name)
        self._notify(ServiceEvent(name=name, change=ServiceChange.REMOVED, old=old, new=None))

    # -- observation ---------------------------------------------------------

    def observe(self, name: str | None, observer: Observer) -> Callable[[], None]:
        """Watch one service name, or all of them when `name` is ``None``."""
        bucket = self._global_observers if name is None else self._observers.setdefault(name, [])
        bucket.append(observer)

        def stop() -> None:
            if observer in bucket:
                bucket.remove(observer)

        return stop

    def _notify(self, event: ServiceEvent) -> None:
        for observer in list(self._observers.get(event.name, ())) + list(self._global_observers):
            try:
                observer(event)
            except Exception:
                logger.exception("service observer failed for %r", event.name)
