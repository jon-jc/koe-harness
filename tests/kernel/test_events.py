"""Event bus: ordering, isolation, and the bail/parallel dispatch modes."""

from __future__ import annotations

import asyncio

from koe.kernel import EventBus


async def test_handlers_run_in_descending_priority() -> None:
    """Normalization must be able to guarantee it runs before persistence."""
    order: list[str] = []
    bus = EventBus()
    bus.subscribe("asr.final", lambda _: order.append("persist"), priority=0)
    bus.subscribe("asr.final", lambda _: order.append("normalize"), priority=100)
    bus.subscribe("asr.final", lambda _: order.append("metrics"), priority=-10)

    await bus.emit("asr.final", {"text": "こんにちは"})

    assert order == ["normalize", "persist", "metrics"]


async def test_equal_priority_preserves_registration_order() -> None:
    order: list[int] = []
    bus = EventBus()
    for i in range(5):
        bus.subscribe("e", lambda _, i=i: order.append(i))

    await bus.emit("e", None)

    assert order == [0, 1, 2, 3, 4]


async def test_a_raising_handler_does_not_stop_the_emit() -> None:
    """A crashing metrics exporter must never drop a caller's audio."""
    delivered: list[str] = []
    bus = EventBus()
    bus.subscribe("audio.frame", lambda _: delivered.append("first"), priority=10)
    bus.subscribe(
        "audio.frame", lambda _: (_ for _ in ()).throw(RuntimeError("exporter down")), priority=5
    )
    bus.subscribe("audio.frame", lambda _: delivered.append("third"), priority=0)

    results = await bus.emit("audio.frame", b"\x00\x01")

    assert delivered == ["first", "third"]
    assert len(results) == 2


async def test_async_and_sync_handlers_interoperate() -> None:
    seen: list[str] = []
    bus = EventBus()

    async def async_handler(payload: str) -> str:
        await asyncio.sleep(0)
        seen.append(f"async:{payload}")
        return "a"

    bus.subscribe("e", async_handler, priority=1)
    bus.subscribe("e", lambda p: seen.append(f"sync:{p}"), priority=0)

    await bus.emit("e", "x")

    assert seen == ["async:x", "sync:x"]


async def test_bail_returns_the_first_non_none_result() -> None:
    """Used where exactly one plugin should win, e.g. a routing override."""
    calls: list[str] = []
    bus = EventBus()

    def abstain(_: object) -> None:
        calls.append("abstain")
        return None

    def decide(_: object) -> str:
        calls.append("decide")
        return "whisper-large-v3"

    def never_reached(_: object) -> str:
        calls.append("never")
        return "other"

    bus.subscribe("route", abstain, priority=10)
    bus.subscribe("route", decide, priority=5)
    bus.subscribe("route", never_reached, priority=0)

    result = await bus.bail("route", {"lang": "ja"})

    assert result == "whisper-large-v3"
    assert calls == ["abstain", "decide"]


async def test_bail_skips_failing_handlers() -> None:
    bus = EventBus()
    bus.subscribe("route", lambda _: (_ for _ in ()).throw(RuntimeError("x")), priority=10)
    bus.subscribe("route", lambda _: "fallback", priority=0)

    assert await bus.bail("route", None) == "fallback"


async def test_predicate_filters_delivery() -> None:
    seen: list[dict] = []
    bus = EventBus()
    bus.subscribe(
        "transcript",
        lambda p: seen.append(p),
        predicate=lambda p: p is not None and p.get("language") == "ja",
    )

    await bus.emit("transcript", {"language": "ja", "text": "はい"})
    await bus.emit("transcript", {"language": "en", "text": "yes"})

    assert [s["text"] for s in seen] == ["はい"]


async def test_unsubscribe_stops_delivery() -> None:
    seen: list[int] = []
    bus = EventBus()
    off = bus.subscribe("e", lambda _: seen.append(1))

    await bus.emit("e", None)
    off()
    await bus.emit("e", None)

    assert seen == [1]
    assert bus.listener_count("e") == 0


async def test_emit_parallel_runs_concurrently() -> None:
    """Independent sinks should not serialize on the realtime path."""
    bus = EventBus(slow_handler_ms=10_000)

    async def slow(_: object) -> str:
        await asyncio.sleep(0.05)
        return "done"

    for _ in range(4):
        bus.subscribe("fanout", slow)

    started = asyncio.get_running_loop().time()
    results = await bus.emit_parallel("fanout", None)
    elapsed = asyncio.get_running_loop().time() - started

    assert results == ["done"] * 4
    # serial would be ~0.2s; concurrent should land far under that
    assert elapsed < 0.15


async def test_emit_parallel_isolates_failures() -> None:
    bus = EventBus()

    async def ok(_: object) -> str:
        return "ok"

    async def bad(_: object) -> str:
        raise RuntimeError("sink down")

    bus.subscribe("fanout", ok)
    bus.subscribe("fanout", bad)
    bus.subscribe("fanout", ok)

    assert await bus.emit_parallel("fanout", None) == ["ok", "ok"]
