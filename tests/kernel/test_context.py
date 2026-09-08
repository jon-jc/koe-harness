"""Context: dependency-gated activation, hot provider swap, spatial filters.

These are the behaviours the voice pipeline actually leans on in production:
swapping an ASR backend without a redeploy, running a plugin only for Japanese
sessions, and guaranteeing a torn-down plugin leaves nothing behind.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from koe.kernel import Context, KoeError, ServiceConflict, plugin


@dataclass
class Utterance:
    language: str
    text: str


# --------------------------------------------------------------------------
# services
# --------------------------------------------------------------------------


def test_service_is_reachable_by_attribute_and_require() -> None:
    ctx = Context()
    ctx.provide("asr", "whisper")

    assert ctx.asr == "whisper"
    assert ctx.require("asr") == "whisper"
    assert "asr" in ctx


def test_unknown_service_raises_a_helpful_error() -> None:
    ctx = Context()
    ctx.provide("asr", object())

    with pytest.raises(AttributeError, match="available: asr"):
        _ = ctx.diarizer


def test_duplicate_registration_requires_explicit_replace() -> None:
    ctx = Context()
    ctx.provide("llm", "claude")

    with pytest.raises(ServiceConflict):
        ctx.provide("llm", "gpt")

    ctx.provide("llm", "gpt", replace=True)
    assert ctx.llm == "gpt"


def test_service_is_revoked_when_its_owning_scope_dies() -> None:
    """A plugin cannot leave a dangling client behind after it unloads."""
    ctx = Context()

    def provider_plugin(c: Context, _config: None) -> None:
        c.provide("asr", "whisper-large-v3")

    fork = ctx.plugin(provider_plugin)
    assert ctx.services.has("asr")

    fork.dispose()
    assert not ctx.services.has("asr")


# --------------------------------------------------------------------------
# temporal composition: dependency-gated activation
# --------------------------------------------------------------------------


def test_plugin_stays_dormant_until_its_dependencies_exist() -> None:
    started: list[str] = []

    @plugin(name="minutes", inject=["llm", "transcript"])
    def minutes_plugin(c: Context, _config: None) -> None:
        started.append("started")

    ctx = Context()
    fork = ctx.plugin(minutes_plugin)

    assert not fork.running
    assert sorted(fork.missing) == ["llm", "transcript"]
    assert started == []

    ctx.provide("llm", "claude")
    assert not fork.running  # still one dependency short
    assert fork.missing == ["transcript"]

    ctx.provide("transcript", "store")
    assert fork.running
    assert started == ["started"]


def test_plugin_stops_when_a_dependency_is_removed() -> None:
    torn_down: list[str] = []

    @plugin(name="dependent", inject=["asr"])
    def dependent(c: Context, _config: None) -> None:
        c.effect("cleanup", lambda: torn_down.append("cleaned"))

    ctx = Context()
    revoke = ctx.provide("asr", "whisper")
    fork = ctx.plugin(dependent)
    assert fork.running

    revoke()

    assert not fork.running
    assert torn_down == ["cleaned"]


def test_replacing_a_provider_hot_restarts_only_its_dependents() -> None:
    """Swap Whisper for a managed vendor without dropping the process.

    The ASR-dependent plugin rebuilds against the new backend; the websocket
    plugin, which depends on nothing, keeps running untouched.
    """
    bound_backends: list[str] = []
    websocket_starts: list[str] = []

    @plugin(name="transcriber", inject=["asr"])
    def transcriber(c: Context, _config: None) -> None:
        bound_backends.append(c.asr)

    @plugin(name="websocket")
    def websocket(c: Context, _config: None) -> None:
        websocket_starts.append("start")

    ctx = Context()
    ctx.provide("asr", "whisper-local")
    ws_fork = ctx.plugin(websocket)
    asr_fork = ctx.plugin(transcriber)

    assert bound_backends == ["whisper-local"]

    ctx.provide("asr", "managed-vendor-v2", replace=True)

    assert bound_backends == ["whisper-local", "managed-vendor-v2"]
    assert asr_fork.restarts == 1
    assert asr_fork.running
    # the unrelated plugin was never touched
    assert websocket_starts == ["start"]
    assert ws_fork.restarts == 0


def test_restart_disposes_the_previous_generation_of_effects() -> None:
    """No effect leaks across a hot restart."""
    events: list[str] = []

    @plugin(name="leaky", inject=["asr"])
    def leaky(c: Context, _config: None) -> None:
        events.append("apply")
        c.effect("teardown", lambda: events.append("teardown"))

    ctx = Context()
    ctx.provide("asr", "v1")
    ctx.plugin(leaky)
    ctx.provide("asr", "v2", replace=True)

    assert events == ["apply", "teardown", "apply"]


def test_optional_dependencies_trigger_restart_but_do_not_gate_startup() -> None:
    generations: list[object] = []

    @plugin(name="observed", inject=["asr"], optional=["cache"])
    def observed(c: Context, _config: None) -> None:
        generations.append(c.get("cache", "no-cache"))

    ctx = Context()
    ctx.provide("asr", "whisper")
    fork = ctx.plugin(observed)

    assert fork.running
    assert generations == ["no-cache"]

    ctx.provide("cache", "redis")
    assert generations == ["no-cache", "redis"]


def test_reload_restarts_every_instance_and_keeps_config() -> None:
    seen_configs: list[str] = []

    @plugin(name="configured")
    def configured(c: Context, config: str) -> None:
        seen_configs.append(config)

    ctx = Context()
    ctx.plugin(configured, "ja")
    ctx.plugin(configured, "en")
    assert seen_configs == ["ja", "en"]

    ctx.reload(configured)

    assert seen_configs == ["ja", "en", "ja", "en"]


def test_non_reusable_plugin_refuses_a_second_instance() -> None:
    @plugin(name="singleton", reusable=False)
    def singleton(c: Context, _config: None) -> None:
        pass

    ctx = Context()
    ctx.plugin(singleton)

    with pytest.raises(KoeError, match="not reusable"):
        ctx.plugin(singleton)


def test_disposing_the_context_unloads_every_plugin() -> None:
    torn: list[str] = []

    @plugin(name="p")
    def p(c: Context, config: str) -> None:
        c.effect("cleanup", lambda: torn.append(config))

    ctx = Context()
    ctx.plugin(p, "a")
    ctx.plugin(p, "b")

    ctx.dispose()

    assert sorted(torn) == ["a", "b"]


def test_a_plugin_failing_to_apply_leaves_no_partial_state() -> None:
    ctx = Context()

    @plugin(name="broken")
    def broken(c: Context, _config: None) -> None:
        c.provide("half-built", object())
        raise RuntimeError("config invalid")

    with pytest.raises(RuntimeError, match="config invalid"):
        ctx.plugin(broken)

    assert not ctx.services.has("half-built")


# --------------------------------------------------------------------------
# spatial composition: filters
# --------------------------------------------------------------------------


async def test_select_scopes_a_subscription_to_matching_payloads() -> None:
    """A Japanese-only plugin simply never sees English audio."""
    ja_seen: list[str] = []
    all_seen: list[str] = []

    ctx = Context()
    ja = ctx.select(lambda u: u.language == "ja")
    ja.on("utterance", lambda u: ja_seen.append(u.text))
    ctx.on("utterance", lambda u: all_seen.append(u.text))

    await ctx.emit("utterance", Utterance("ja", "承知しました"))
    await ctx.emit("utterance", Utterance("en", "understood"))

    assert ja_seen == ["承知しました"]
    assert all_seen == ["承知しました", "understood"]


async def test_filters_compose_by_intersection() -> None:
    seen: list[str] = []

    ctx = Context()
    narrow = ctx.select(lambda u: u.language == "ja").intersect(lambda u: len(u.text) > 3)
    narrow.on("utterance", lambda u: seen.append(u.text))

    await ctx.emit("utterance", Utterance("ja", "はい"))
    await ctx.emit("utterance", Utterance("ja", "承知しました"))
    await ctx.emit("utterance", Utterance("en", "understood"))

    assert seen == ["承知しました"]


async def test_union_and_exclude() -> None:
    union_seen: list[str] = []
    exclude_seen: list[str] = []

    ctx = Context()
    ctx.select(lambda u: u.language == "ja").union(lambda u: u.language == "en").on(
        "u", lambda u: union_seen.append(u.language)
    )
    ctx.exclude(lambda u: u.language == "en").on("u", lambda u: exclude_seen.append(u.language))

    for lang in ("ja", "en", "ko"):
        await ctx.emit("u", Utterance(lang, "x"))

    assert union_seen == ["ja", "en"]
    assert exclude_seen == ["ja", "ko"]


async def test_a_plugin_loaded_on_a_filtered_context_inherits_the_filter() -> None:
    """Spatial and temporal composition stack."""
    seen: list[str] = []

    @plugin(name="ja_minutes", inject=["llm"])
    def ja_minutes(c: Context, _config: None) -> None:
        c.on("utterance", lambda u: seen.append(u.text))

    ctx = Context()
    ja = ctx.select(lambda u: u.language == "ja")
    fork = ja.plugin(ja_minutes)

    assert not fork.running  # gated on the LLM service
    ctx.provide("llm", "claude")
    assert fork.running

    await ctx.emit("utterance", Utterance("ja", "議事録を作成"))
    await ctx.emit("utterance", Utterance("en", "make minutes"))

    assert seen == ["議事録を作成"]


async def test_a_raising_filter_is_treated_as_non_matching() -> None:
    seen: list[str] = []
    ctx = Context()
    ctx.select(lambda u: u.missing_attribute == "x").on("u", lambda u: seen.append("hit"))

    await ctx.emit("u", Utterance("ja", "x"))

    assert seen == []


# --------------------------------------------------------------------------
# events through the context
# --------------------------------------------------------------------------


async def test_once_fires_a_single_time() -> None:
    seen: list[int] = []
    ctx = Context()
    ctx.once("tick", lambda _: seen.append(1))

    await ctx.emit("tick", None)
    await ctx.emit("tick", None)

    assert seen == [1]


async def test_on_works_as_a_decorator() -> None:
    seen: list[str] = []
    ctx = Context()

    @ctx.on("greet")
    def handler(name: str) -> None:
        seen.append(name)

    await ctx.emit("greet", "koe")
    assert seen == ["koe"]


async def test_subscriptions_die_with_their_plugin() -> None:
    seen: list[str] = []

    @plugin(name="listener")
    def listener(c: Context, _config: None) -> None:
        c.on("utterance", lambda u: seen.append(u.text))

    ctx = Context()
    fork = ctx.plugin(listener)

    await ctx.emit("utterance", Utterance("ja", "one"))
    fork.dispose()
    await ctx.emit("utterance", Utterance("ja", "two"))

    assert seen == ["one"]


# --------------------------------------------------------------------------
# async plugins
# --------------------------------------------------------------------------


async def test_async_plugin_applies_via_plugin_async() -> None:
    started: list[str] = []

    @plugin(name="async_provider")
    async def async_provider(c: Context, _config: None) -> None:
        await asyncio.sleep(0)
        c.provide("connection", "open")
        started.append("ready")

    ctx = Context()
    fork = await ctx.plugin_async(async_provider)

    assert started == ["ready"]
    assert ctx.connection == "open"
    fork.dispose()
    assert not ctx.services.has("connection")


def test_async_plugin_outside_a_loop_fails_loudly() -> None:
    @plugin(name="async_provider")
    async def async_provider(c: Context, _config: None) -> None:
        pass

    ctx = Context()
    with pytest.raises(KoeError, match="running event loop"):
        ctx.plugin(async_provider)


# --------------------------------------------------------------------------
# plugin shapes
# --------------------------------------------------------------------------


def test_class_style_plugins_are_supported() -> None:
    seen: list[str] = []

    class DiarizerPlugin:
        name = "diarizer"
        inject = ("asr",)

        def apply(self, c: Context, config: str) -> None:
            seen.append(f"{config}:{c.asr}")

    ctx = Context()
    ctx.provide("asr", "whisper")
    ctx.plugin(DiarizerPlugin, "pyannote")

    assert seen == ["pyannote:whisper"]


def test_plugin_introspection_lists_live_instances() -> None:
    @plugin(name="p")
    def p(c: Context, _config: None) -> None:
        pass

    ctx = Context()
    ctx.plugin(p)
    ctx.plugin(p)

    mains = list(ctx.plugins())
    assert len(mains) == 1
    assert len(mains[0].forks) == 2
