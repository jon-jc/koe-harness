"""The context: koe's composition root.

A :class:`Context` bundles three things a plugin needs -- a service registry, an
event bus, and a scope that owns whatever it creates -- and hands out *derived*
contexts that share the first two while narrowing the third.

That narrowing happens on two axes, which together are what make the harness
composable rather than merely modular:

**Temporal.** ``ctx.plugin(p)`` returns a :class:`ForkScope`. The fork watches
the services listed in ``p.inject`` and keeps the plugin alive exactly while
they all exist. Register an ASR backend and every ASR-dependent plugin starts;
replace it and they are torn down and rebuilt against the new one; remove it
and they stop. Nothing polls, and no plugin has to write a reconnect path.

**Spatial.** ``ctx.select(pred)`` returns a context whose subscriptions only
fire for payloads matching ``pred``. A plugin loaded on
``ctx.select(lambda e: e.language == "ja")`` is, from its own point of view, a
normal plugin -- it simply never sees English audio.

The two compose. A Japanese-only summarizer that requires an LLM is::

    ja = ctx.select(lambda ev: ev.language == "ja")
    ja.plugin(minutes_plugin, MinutesConfig(style="keigo"))

which is live only for Japanese sessions, and only while an LLM service is
registered.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterator
from typing import Any

from koe.kernel import plugin as plugin_mod
from koe.kernel.errors import KoeError, ServiceNotFound
from koe.kernel.events import EventBus, Handler
from koe.kernel.plugin import PluginSpec
from koe.kernel.scope import Scope
from koe.kernel.services import ServiceChange, ServiceEvent, ServiceRegistry

logger = logging.getLogger(__name__)

Predicate = Callable[[Any], bool]


async def _drive(awaitable: Any) -> Any:
    """Wrap an arbitrary awaitable so it can be scheduled as a task."""
    return await awaitable


class ForkScope:
    """One live instance of a plugin.

    Owns the plugin's effects and decides, from service availability, whether
    the plugin should currently be running.
    """

    def __init__(
        self,
        ctx: Context,
        spec: PluginSpec,
        config: Any,
        parent_scope: Scope,
    ) -> None:
        self._ctx = ctx
        self.spec = spec
        self.config = config
        self._scope = parent_scope.fork(spec.name)
        self._effects: Scope | None = None
        self._running = False
        self._disposed = False
        self._task: asyncio.Task[Any] | None = None
        self._unwatch: list[Callable[[], None]] = []
        self.restarts = 0

        watched = set(spec.inject) | set(spec.optional)
        for name in watched:
            self._unwatch.append(ctx.services.observe(name, self._on_service_change))
        self._scope.collect("unwatch-services", self._detach_watchers)

        self._evaluate(reason="load")

    # -- state ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def missing(self) -> list[str]:
        """Injected services that are not currently available."""
        return [n for n in self.spec.inject if not self._ctx.services.has(n)]

    def _detach_watchers(self) -> None:
        for stop in self._unwatch:
            stop()
        self._unwatch.clear()

    # -- reactivity ----------------------------------------------------------

    def _on_service_change(self, event: ServiceEvent) -> None:
        """Translate a service change into a start, stop, or rebuild.

        While the plugin is running, any change to a watched service means the
        view of the world it captured at apply time is now stale, so it is
        rebuilt -- *except* the one case where a required dependency vanished,
        which means it must stop instead of restarting into an unsatisfiable
        state.
        """
        if self._disposed:
            return

        reason = f"service:{event.name}:{event.change.value}"

        if self._running:
            required_gone = event.change is ServiceChange.REMOVED and event.name in self.spec.inject
            if required_gone:
                self._evaluate(reason=reason)  # -> stops, dependency is unsatisfiable
                return
            logger.info("restarting plugin %r: %s", self.spec.name, reason)
            self.restart(reason=reason)
            return

        self._evaluate(reason=reason)

    def _evaluate(self, *, reason: str) -> None:
        if self._disposed:
            return
        ready = not self.missing
        if ready and not self._running:
            self._start(reason=reason)
        elif not ready and self._running:
            logger.info("stopping plugin %r: missing %s (%s)", self.spec.name, self.missing, reason)
            self._stop()

    # -- lifecycle -----------------------------------------------------------

    def _start(self, *, reason: str) -> None:
        self._effects = self._scope.fork("effects")
        plugin_ctx = self._ctx._derive(scope=self._effects, plugin_spec=self.spec)
        self._running = True
        logger.debug("starting plugin %r (%s)", self.spec.name, reason)
        try:
            result = (
                self.spec.apply(plugin_ctx, self.config)
                if self.spec.takes_config
                else self.spec.apply(plugin_ctx)
            )
        except Exception:
            self._running = False
            self._effects.dispose()
            self._effects = None
            logger.exception("plugin %r failed during apply", self.spec.name)
            raise

        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                # close the orphaned coroutine so it does not surface later as a
                # bare "never awaited" warning far from the actual mistake
                if inspect.iscoroutine(result):
                    result.close()
                self._running = False
                self._effects.dispose()
                self._effects = None
                raise KoeError(
                    f"plugin {self.spec.name!r} has an async apply() and needs a running event "
                    "loop; load it with `await ctx.plugin_async(...)`"
                ) from exc
            task: asyncio.Task[Any] = loop.create_task(
                _drive(result), name=f"koe.plugin.{self.spec.name}"
            )
            self._task = task
            self._effects.collect("cancel-apply-task", task.cancel)

    def _stop(self) -> None:
        self._running = False
        self._task = None
        if self._effects is not None:
            effects, self._effects = self._effects, None
            effects.dispose()

    def restart(self, *, reason: str = "manual") -> None:
        """Tear down the plugin's effects and re-apply it."""
        if self._disposed:
            return
        was_running = self._running
        self._stop()
        self.restarts += 1
        if was_running:
            self._evaluate(reason=reason)

    async def wait_ready(self) -> None:
        """Await an async plugin's ``apply`` coroutine."""
        if self._task is not None:
            await self._task

    def dispose(self) -> None:
        """Permanently unload this plugin instance."""
        if self._disposed:
            return
        self._disposed = True
        self._stop()
        self._scope.dispose()

    def __repr__(self) -> str:
        state = "disposed" if self._disposed else ("running" if self._running else "waiting")
        extra = f" missing={self.missing}" if self.missing else ""
        return f"<ForkScope {self.spec.name!r} {state}{extra}>"


class MainScope:
    """All live instances of one plugin, grouped for reload and introspection."""

    def __init__(self, spec: PluginSpec) -> None:
        self.spec = spec
        self.forks: list[ForkScope] = []

    def add(self, fork: ForkScope) -> None:
        self.forks.append(fork)

    def prune(self) -> None:
        self.forks = [f for f in self.forks if not f.disposed]

    def __repr__(self) -> str:
        return f"<MainScope {self.spec.name!r} forks={len(self.forks)}>"


class Context:
    """A composable handle onto services, events and lifetime."""

    def __init__(self, *, name: str = "root", slow_handler_ms: float = 50.0) -> None:
        self._root: Context = self
        self.services = ServiceRegistry()
        self.bus = EventBus(slow_handler_ms=slow_handler_ms)
        self._registry: dict[Any, MainScope] = {}
        self._scope = Scope(name)
        self._filter: Predicate | None = None
        self._plugin_spec: PluginSpec | None = None

    # -- derivation ----------------------------------------------------------

    def _derive(
        self,
        *,
        scope: Scope,
        filter_: Predicate | None = None,
        plugin_spec: PluginSpec | None = None,
    ) -> Context:
        child = object.__new__(Context)
        child._root = self._root
        child.services = self._root.services
        child.bus = self._root.bus
        child._registry = self._root._registry
        child._scope = scope
        child._filter = filter_ if filter_ is not None else self._filter
        child._plugin_spec = plugin_spec or self._plugin_spec
        return child

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def is_root(self) -> bool:
        return self._root is self

    # -- spatial composition -------------------------------------------------

    def select(self, predicate: Predicate) -> Context:
        """Narrow this context to payloads matching `predicate`."""
        return self.intersect(predicate)

    def intersect(self, predicate: Predicate) -> Context:
        base = self._filter
        if base is None:
            combined: Predicate = predicate
        else:

            def combined(payload: Any) -> bool:
                return bool(base(payload) and predicate(payload))

        return self._derive(scope=self._scope, filter_=combined)

    def union(self, predicate: Predicate) -> Context:
        base = self._filter
        if base is None:
            combined: Predicate = predicate
        else:

            def combined(payload: Any) -> bool:
                return bool(base(payload) or predicate(payload))

        return self._derive(scope=self._scope, filter_=combined)

    def exclude(self, predicate: Predicate) -> Context:
        base = self._filter

        if base is None:

            def combined(payload: Any) -> bool:
                return not predicate(payload)

        else:

            def combined(payload: Any) -> bool:
                return bool(base(payload) and not predicate(payload))

        return self._derive(scope=self._scope, filter_=combined)

    def matches(self, payload: Any) -> bool:
        """Whether `payload` passes this context's filter."""
        if self._filter is None:
            return True
        try:
            return bool(self._filter(payload))
        except Exception:
            logger.exception("context filter raised; treating payload as non-matching")
            return False

    # -- services ------------------------------------------------------------

    def provide(self, name: str, value: Any, *, replace: bool = False) -> Callable[[], None]:
        """Register a service owned by this context's scope."""
        self.services.set(name, value, owner=self._scope.path, replace=replace)

        def revoke() -> None:
            if self.services.has(name) and self.services.get(name) is value:
                self.services.delete(name)

        self._scope.collect(f"service:{name}", revoke)
        return revoke

    def get(self, name: str, default: Any = None) -> Any:
        return self.services.get(name, default)

    def require(self, name: str) -> Any:
        """Fetch a service or raise :class:`ServiceNotFound`."""
        return self.services.get(name)

    def __getattr__(self, name: str) -> Any:
        # only reached for names not resolved normally
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._root.services.get(name)
        except ServiceNotFound as exc:
            raise AttributeError(str(exc)) from exc

    def __contains__(self, name: str) -> bool:
        return self.services.has(name)

    # -- events --------------------------------------------------------------

    def on(self, event: str, handler: Handler | None = None, *, priority: int = 0) -> Any:
        """Subscribe to `event`, honouring this context's spatial filter.

        Usable directly or as a decorator.
        """

        def register(fn: Handler) -> Handler:
            predicate = self.matches if self._filter is not None else None
            unsubscribe = self.bus.subscribe(event, fn, priority=priority, predicate=predicate)
            self._scope.collect(f"on:{event}", unsubscribe)
            return fn

        if handler is None:
            return register
        register(handler)
        return handler

    def once(self, event: str, handler: Handler) -> Callable[[], None]:
        """Subscribe for a single delivery."""
        state = {"done": False}
        holder: dict[str, Callable[[], None]] = {}

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if state["done"]:
                return None
            state["done"] = True
            holder["off"]()
            result = handler(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        predicate = self.matches if self._filter is not None else None
        off = self.bus.subscribe(event, wrapper, predicate=predicate)
        holder["off"] = off
        self._scope.collect(f"once:{event}", off)
        return off

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        return await self.bus.emit(event, *args, **kwargs)

    async def emit_parallel(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        return await self.bus.emit_parallel(event, *args, **kwargs)

    async def bail(self, event: str, *args: Any, **kwargs: Any) -> Any:
        return await self.bus.bail(event, *args, **kwargs)

    # -- effects -------------------------------------------------------------

    def effect(self, label: str, teardown: Callable[[], Any]) -> None:
        """Attach an arbitrary teardown to this context's lifetime."""
        self._scope.collect(label, teardown)

    # -- plugins -------------------------------------------------------------

    def plugin(self, source: Any, config: Any = None) -> ForkScope:
        """Load a plugin into this context and return its fork."""
        spec = plugin_mod.normalize(source)
        key = spec.origin if spec.origin is not None else spec.apply
        main = self._registry.get(key)
        if main is None:
            main = MainScope(spec)
            self._registry[key] = main
        elif not spec.reusable and any(not f.disposed for f in main.forks):
            raise KoeError(f"plugin {spec.name!r} is not reusable and is already loaded")

        fork = ForkScope(self, spec, config, self._scope)
        main.add(fork)
        self._scope.collect(f"plugin:{spec.name}", fork.dispose)
        return fork

    async def plugin_async(self, source: Any, config: Any = None) -> ForkScope:
        """Load a plugin and await its ``apply`` if it is a coroutine."""
        fork = self.plugin(source, config)
        await fork.wait_ready()
        return fork

    def reload(self, source: Any) -> list[ForkScope]:
        """Restart every live instance of `source`, preserving configs."""
        spec = plugin_mod.normalize(source)
        key = spec.origin if spec.origin is not None else spec.apply
        main = self._registry.get(key)
        if main is None:
            return []
        main.prune()
        for fork in main.forks:
            fork.restart(reason="reload")
        return list(main.forks)

    def plugins(self) -> Iterator[MainScope]:
        for main in list(self._registry.values()):
            main.prune()
            if main.forks:
                yield main

    # -- teardown ------------------------------------------------------------

    def dispose(self) -> None:
        """Tear down everything registered on this context."""
        self._scope.dispose()

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.dispose()

    def __repr__(self) -> str:
        tag = f" plugin={self._plugin_spec.name!r}" if self._plugin_spec else ""
        filt = " filtered" if self._filter is not None else ""
        return f"<Context {self._scope.path!r}{tag}{filt} services={len(self.services.names())}>"
