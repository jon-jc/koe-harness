"""Scoped resource ownership.

A :class:`Scope` is a bag of teardown callbacks with a parent/child tree. It is
the only mechanism in koe for owning a resource, and it exists because the
realtime path creates and destroys a lot of them: every call leg opens sockets,
decoder sessions, provider clients and metric timers, and every one of those
must go away when the call ends -- including when it ends by crashing.

Two rules make that reliable:

* **Children die before parents.** A decoder cannot outlive the session that
  owns its audio buffer.
* **Disposal always completes.** One failing callback never prevents the rest
  from running; failures are collected and raised together as
  :class:`~koe.kernel.errors.DisposalError`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from koe.kernel.errors import DisposalError, ScopeDisposed

logger = logging.getLogger(__name__)

Teardown = Callable[[], Any]


class Scope:
    """An owner of teardown callbacks, arranged in a tree."""

    __slots__ = ("_active", "_children", "_teardowns", "name", "parent")

    def __init__(self, name: str, parent: Scope | None = None) -> None:
        self.name = name
        self.parent = parent
        self._teardowns: list[tuple[str, Teardown]] = []
        self._children: list[Scope] = []
        self._active = True

    # -- state ---------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether this scope can still accept work."""
        return self._active

    @property
    def path(self) -> str:
        """Dotted path from the root, useful in logs and traces."""
        if self.parent is None:
            return self.name
        return f"{self.parent.path}.{self.name}"

    def _check(self) -> None:
        if not self._active:
            raise ScopeDisposed(self.path)

    # -- registration --------------------------------------------------------

    def collect(self, label: str, teardown: Teardown) -> None:
        """Register `teardown` to run when this scope is disposed."""
        self._check()
        self._teardowns.append((label, teardown))

    def fork(self, name: str) -> Scope:
        """Create a child scope that is disposed before this one."""
        self._check()
        child = Scope(name, parent=self)
        self._children.append(child)
        return child

    # -- teardown ------------------------------------------------------------

    def dispose(self) -> None:
        """Tear down this scope and everything beneath it.

        Idempotent. Children first, then this scope's own callbacks in reverse
        registration order, so teardown mirrors construction.
        """
        if not self._active:
            return
        self._active = False

        failures: list[tuple[str, BaseException]] = []

        for child in reversed(self._children):
            try:
                child.dispose()
            except DisposalError as exc:
                failures.extend(exc.failures)
            except BaseException as exc:  # noqa: BLE001 - teardown must not propagate early
                failures.append((f"{child.name}(scope)", exc))
        self._children.clear()

        for label, teardown in reversed(self._teardowns):
            try:
                teardown()
            except BaseException as exc:
                logger.warning("teardown %r failed in scope %s", label, self.path, exc_info=exc)
                failures.append((label, exc))
        self._teardowns.clear()

        if self.parent is not None and self in self.parent._children:
            self.parent._children.remove(self)

        if failures:
            raise DisposalError(self.path, failures)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> Scope:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.dispose()

    def __repr__(self) -> str:
        state = "active" if self._active else "disposed"
        return f"<Scope {self.path!r} {state} teardowns={len(self._teardowns)} children={len(self._children)}>"
