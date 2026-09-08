"""An async event bus with priorities, predicates and back-pressure awareness.

The realtime pipeline is a fan-out of small events -- `audio.frame`,
`asr.partial`, `asr.final`, `turn.end` -- consumed by stages that must not
block each other. This bus is deliberately small but has three properties the
voice path needs:

* **Handlers cannot take the pipeline down.** A raising subscriber is logged
  and isolated; the emit still reaches everyone else. A crashing metrics
  exporter must never drop a caller's audio.
* **Ordering is explicit.** Handlers run in descending `priority`, so
  normalization can be guaranteed to run before persistence.
* **Slow handlers are visible.** Anything exceeding `slow_handler_ms` is
  logged with its event name, because in a realtime system a slow subscriber
  is a latency bug that would otherwise hide.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[..., Any]
Predicate = Callable[[Any], bool]


@dataclass(order=True)
class _Subscription:
    # negated priority first so heapless sort ascending == priority descending
    sort_key: tuple[int, int] = field(compare=True)
    event: str = field(compare=False, default="")
    handler: Handler = field(compare=False, default=lambda: None)
    predicate: Predicate | None = field(compare=False, default=None)
    is_async: bool = field(compare=False, default=False)
    label: str = field(compare=False, default="")
    active: bool = field(compare=False, default=True)


class EventBus:
    """Fan-out dispatcher shared by every :class:`~koe.kernel.context.Context`."""

    def __init__(self, *, slow_handler_ms: float = 50.0) -> None:
        self._subs: dict[str, list[_Subscription]] = {}
        self._seq = 0
        self._slow_handler_ms = slow_handler_ms

    # -- subscription --------------------------------------------------------

    def subscribe(
        self,
        event: str,
        handler: Handler,
        *,
        priority: int = 0,
        predicate: Predicate | None = None,
    ) -> Callable[[], None]:
        """Register `handler` for `event`; returns a function that unsubscribes."""
        self._seq += 1
        sub = _Subscription(
            sort_key=(-priority, self._seq),
            event=event,
            handler=handler,
            predicate=predicate,
            is_async=inspect.iscoroutinefunction(handler),
            label=getattr(handler, "__qualname__", repr(handler)),
        )
        bucket = self._subs.setdefault(event, [])
        bucket.append(sub)
        bucket.sort()

        def unsubscribe() -> None:
            sub.active = False
            current = self._subs.get(event)
            if current and sub in current:
                current.remove(sub)
            if current is not None and not current:
                self._subs.pop(event, None)

        return unsubscribe

    def listener_count(self, event: str) -> int:
        return len(self._subs.get(event, ()))

    # -- dispatch ------------------------------------------------------------

    def _eligible(self, event: str, payload: Any) -> list[_Subscription]:
        out = []
        for sub in tuple(self._subs.get(event, ())):
            if not sub.active:
                continue
            if sub.predicate is not None and not sub.predicate(payload):
                continue
            out.append(sub)
        return out

    async def _invoke(
        self, sub: _Subscription, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        started = time.perf_counter()
        try:
            result = sub.handler(*args, **kwargs)
            if sub.is_async or inspect.isawaitable(result):
                result = await result
            return result
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms > self._slow_handler_ms:
                logger.warning(
                    "slow handler %s on %r took %.1fms (budget %.0fms)",
                    sub.label,
                    sub.event,
                    elapsed_ms,
                    self._slow_handler_ms,
                )

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Run every eligible handler in priority order; isolate failures."""
        payload = args[0] if args else None
        results: list[Any] = []
        for sub in self._eligible(event, payload):
            try:
                results.append(await self._invoke(sub, args, kwargs))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("handler %s failed on event %r", sub.label, event, exc_info=exc)
        return results

    async def bail(self, event: str, *args: Any, **kwargs: Any) -> Any:
        """Run handlers in priority order and return the first non-``None`` result.

        Used where exactly one plugin should win -- routing overrides, cache
        lookups, guardrail vetoes.
        """
        payload = args[0] if args else None
        for sub in self._eligible(event, payload):
            try:
                result = await self._invoke(sub, args, kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("handler %s failed on event %r", sub.label, event, exc_info=exc)
                continue
            if result is not None:
                return result
        return None

    async def emit_parallel(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Fan out concurrently, ignoring priority.

        For genuinely independent sinks (metrics, transcript persistence,
        websocket broadcast) where serializing would add avoidable latency.
        """
        payload = args[0] if args else None
        subs = self._eligible(event, payload)
        if not subs:
            return []
        settled = await asyncio.gather(
            *(self._invoke(sub, args, kwargs) for sub in subs),
            return_exceptions=True,
        )
        out: list[Any] = []
        for sub, result in zip(subs, settled, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                logger.exception("handler %s failed on event %r", sub.label, event, exc_info=result)
                continue
            out.append(result)
        return out
