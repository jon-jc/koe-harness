"""Plugin declaration and normalization.

A plugin is anything that can attach behaviour to a :class:`Context`. koe
accepts three shapes so that trivial plugins stay trivial:

1. a plain function ``def p(ctx, config): ...``
2. a decorated function ``@plugin(name="asr", inject=["audio"])``
3. a class exposing ``apply(self, ctx, config)`` plus class-level metadata

All three are normalized to a :class:`PluginSpec`, which is the only shape the
registry deals with.

`inject` is the interesting field. It lists services the plugin cannot run
without. The kernel uses it to decide *when* a plugin is allowed to be alive:
a plugin declaring ``inject=["asr"]`` does not run at load time, it runs the
moment an ASR service appears -- and is torn down and rebuilt if that service
is later replaced. That is what lets a running server swap Whisper for a
managed ASR vendor without dropping the process.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from koe.kernel.errors import KoeError

F = TypeVar("F", bound=Callable[..., Any])


@runtime_checkable
class PluginObject(Protocol):
    """A class-style plugin."""

    def apply(self, ctx: Any, config: Any) -> Any: ...


@dataclass(slots=True)
class PluginSpec:
    """Normalized plugin metadata."""

    name: str
    apply: Callable[..., Any]
    inject: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    reusable: bool = True
    origin: Any = None

    @property
    def takes_config(self) -> bool:
        try:
            sig = inspect.signature(self.apply)
        except (TypeError, ValueError):
            return True
        return len(sig.parameters) >= 2

    def __repr__(self) -> str:
        deps = "+".join(self.inject) or "-"
        return f"<PluginSpec {self.name!r} inject={deps}>"


def plugin(
    _fn: F | None = None,
    *,
    name: str | None = None,
    inject: Sequence[str] = (),
    optional: Sequence[str] = (),
    reusable: bool = True,
) -> Any:
    """Decorate a function as a koe plugin.

    >>> @plugin(name="minutes", inject=["llm", "transcript"])
    ... def minutes_plugin(ctx, config):
    ...     ctx.on("session.end", ...)
    """

    def wrap(fn: F) -> F:
        fn.__koe_plugin__ = {  # type: ignore[attr-defined]
            "name": name or getattr(fn, "__name__", "anonymous"),
            "inject": tuple(inject),
            "optional": tuple(optional),
            "reusable": reusable,
        }
        return fn

    if _fn is not None:
        return wrap(_fn)
    return wrap


def _meta(obj: Any, key: str, default: Any) -> Any:
    declared = getattr(obj, "__koe_plugin__", None)
    if isinstance(declared, dict) and key in declared:
        return declared[key]
    return getattr(obj, key, default)


def normalize(source: Any) -> PluginSpec:
    """Coerce any accepted plugin shape into a :class:`PluginSpec`."""
    if isinstance(source, PluginSpec):
        return source

    inject = tuple(_meta(source, "inject", ()))
    optional = tuple(_meta(source, "optional", ()))
    reusable = bool(_meta(source, "reusable", True))

    # class-style: instantiate lazily via apply
    if inspect.isclass(source):
        if not hasattr(source, "apply"):
            raise KoeError(f"plugin class {source.__name__!r} must define apply(self, ctx, config)")
        name = _meta(source, "name", source.__name__)

        def apply_class(ctx: Any, config: Any, _cls: Any = source) -> Any:
            instance = _cls()
            return instance.apply(ctx, config)

        return PluginSpec(
            name=str(name),
            apply=apply_class,
            inject=inject,
            optional=optional,
            reusable=reusable,
            origin=source,
        )

    # instance exposing .apply
    if hasattr(source, "apply") and not inspect.isfunction(source):
        name = _meta(source, "name", type(source).__name__)
        return PluginSpec(
            name=str(name),
            apply=source.apply,
            inject=inject,
            optional=optional,
            reusable=reusable,
            origin=source,
        )

    # plain / decorated callable
    if callable(source):
        name = _meta(source, "name", getattr(source, "__name__", "anonymous"))
        return PluginSpec(
            name=str(name),
            apply=source,
            inject=inject,
            optional=optional,
            reusable=reusable,
            origin=source,
        )

    raise KoeError(
        f"{source!r} is not a valid plugin (need a callable, class, or object with .apply)"
    )
