"""Kernel-level exceptions.

Every error the harness raises derives from :class:`KoeError` so callers can
draw a single boundary around the library.
"""

from __future__ import annotations


class KoeError(Exception):
    """Base class for every error raised by koe."""


class ServiceNotFound(KoeError):
    """A plugin asked for a service that is not registered."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        known = ", ".join(sorted(available)) or "<none>"
        super().__init__(f"service {name!r} is not registered (available: {known})")


class ServiceConflict(KoeError):
    """Two plugins tried to claim the same service name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"service {name!r} is already registered; pass replace=True to override it"
        )


class ScopeDisposed(KoeError):
    """An operation was attempted on a scope that has already been torn down."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(f"scope {scope!r} has been disposed and can no longer be used")


class DisposalError(KoeError):
    """One or more teardown callbacks raised.

    Disposal always runs to completion: every callback is attempted even if an
    earlier one fails, and the failures are aggregated here. A leaked socket in
    one plugin must not strand the sockets of its siblings.
    """

    def __init__(self, scope: str, failures: list[tuple[str, BaseException]]) -> None:
        self.scope = scope
        self.failures = failures
        detail = "; ".join(f"{label}: {exc!r}" for label, exc in failures)
        super().__init__(
            f"{len(failures)} teardown callback(s) failed in scope {scope!r}: {detail}"
        )
