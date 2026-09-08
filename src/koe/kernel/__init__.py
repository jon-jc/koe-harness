"""koe kernel -- composable plugins, scoped lifetimes, reactive services.

The kernel knows nothing about audio. It provides the four primitives every
other package in koe is built from:

* :class:`~koe.kernel.context.Context` -- composition root, derived along a
  spatial (filter) and temporal (dependency) axis.
* :class:`~koe.kernel.scope.Scope` -- tree-structured ownership of teardown.
* :class:`~koe.kernel.services.ServiceRegistry` -- reactive dependency
  injection that announces adds, replacements and removals.
* :class:`~koe.kernel.events.EventBus` -- prioritized async fan-out with
  failure isolation and slow-handler detection.
"""

from koe.kernel.context import Context, ForkScope, MainScope
from koe.kernel.errors import (
    DisposalError,
    KoeError,
    ScopeDisposed,
    ServiceConflict,
    ServiceNotFound,
)
from koe.kernel.events import EventBus
from koe.kernel.plugin import PluginSpec, normalize, plugin
from koe.kernel.scope import Scope
from koe.kernel.services import ServiceChange, ServiceEvent, ServiceRegistry

__all__ = [
    "Context",
    "DisposalError",
    "EventBus",
    "ForkScope",
    "KoeError",
    "MainScope",
    "PluginSpec",
    "Scope",
    "ScopeDisposed",
    "ServiceChange",
    "ServiceConflict",
    "ServiceEvent",
    "ServiceNotFound",
    "ServiceRegistry",
    "normalize",
    "plugin",
]
