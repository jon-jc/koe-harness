"""The tool registry and its guarded execution pipeline.

A tool is the unit an LLM can actually *do* something with, and the registry is
where the harness decides what "actually" means. Four rules shape it, each one
a position rather than a default.

**The model-facing schema is an allowlist.** ``schemas()`` copies name,
description and parameters and nothing else. Host-only fields — timeouts,
approval requirements, concurrency metadata — exist on the same object because
that is where they belong for the code that runs the tool, and the one place
they must never appear is a request. A denylist here fails open: someone adds a
field, forgets the filter, and it ships to the vendor. An allowlist fails
closed.

**A failed tool is a result, not an exception.** A model that asks to read a
missing file should be told the file is missing and given the chance to try
another path. Raising through the agent loop instead ends the turn and throws
away everything the model had established. So every failure — a bad argument, a
policy refusal, a timeout, a crash in the tool body — comes back as a
:class:`ToolResult` with a stable code the model and the UI can both route on.

**Approval is a plugin, not a flag the tool checks.** Tools declare that they
are dangerous; they do not decide what happens about it. A policy listening on
``tools/pre-execute`` decides. Removing the policy degrades to unguarded
execution rather than breaking every tool, which is the property that lets a
headless deployment and a desktop app share one tool set.

**The pipeline is events, so it is extensible without being edited.** Three
phases — ``tools/pre-execute``, ``tools/execute``, ``tools/post-execute`` —
are the seams where a plugin adds auditing, redaction, rate limiting or a
sandbox without this module knowing those things exist.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

#: Never send a tool result larger than this to a model. A tool that returns a
#: 40 MB file does not produce a better answer, it produces a context overflow
#: and a bill. Truncation is visible in the result so the model can narrow.
MAX_RESULT_CHARS = 60_000


class ToolError(StrEnum):
    """Stable, machine-routable failure codes.

    Strings rather than an opaque enum because they cross a JSON boundary to
    both the model and the browser, and both need to branch on them.
    """

    NO_TOOL = "no_tool"
    BAD_ARGUMENTS = "bad_arguments"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ToolRun:
    """Identity and cancellation for one call.

    Passed to the tool body rather than assembled by it, so a tool cannot
    invent an identity or ignore a cancellation it was never handed.
    """

    tool: str
    call_id: str
    #: Who is running this. Terminal sessions and file writes are fenced to it.
    owner: str = "default"
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Set when the caller gives up. Long-running tools must observe it.
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    def raise_if_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise asyncio.CancelledError


@dataclass(slots=True)
class ToolResult:
    """The outcome of one call, in the shape both the model and the UI read."""

    tool: str
    call_id: str
    ok: bool
    #: Text handed back to the model. Always a string: a model reads text.
    content: str = ""
    #: The structured value, for a UI that can render more than text.
    value: Any = None
    error: ToolError | None = None
    detail: str = ""
    duration_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "call_id": self.call_id,
            "ok": self.ok,
            "content": self.content,
            "value": self.value,
            "error": self.error.value if self.error else None,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
            "truncated": self.truncated,
        }


ToolBody = Callable[[dict[str, Any], ToolRun], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    """A registered tool: what the model sees, plus what only the host sees."""

    name: str
    description: str
    #: JSON Schema for the arguments. Sent to the model verbatim.
    parameters: dict[str, Any]
    execute: ToolBody

    # -- host-only, never model-visible -----------------------------------
    #: Cooperative deadline. The body must observe `run.cancelled` for this to
    #: mean anything; the registry can stop waiting but cannot kill the work.
    timeout_s: float | None = None
    #: Declares that this tool changes something outside the process. The
    #: registry does not act on it — a policy plugin does.
    dangerous: bool = False
    #: Which plugin contributed it, for the settings UI and for teardown.
    source: str = "builtin"

    def schema(self) -> dict[str, Any]:
        """The model-facing projection. Allowlist, deliberately."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class Denial:
    """A refusal from `tools/pre-execute`."""

    reason: str
    code: ToolError = ToolError.DENIED


class ToolRegistry:
    """Holds tool definitions and runs calls through the guarded pipeline."""

    def __init__(self, ctx: Any = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._ctx = ctx

    # -- registration ------------------------------------------------------

    def register(self, spec: ToolSpec) -> Callable[[], None]:
        """Add a tool and return its disposer.

        Returning the disposer rather than exposing ``unregister(name)`` is
        what makes a registration an *effect*: the plugin that added the tool
        holds the only handle that removes it, so unloading the plugin cannot
        leave a tool behind pointing at code that is gone.
        """
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

        def dispose() -> None:
            # Guarded because a scope can be torn down twice, and because a
            # replacement may already own the name.
            if self._tools.get(spec.name) is spec:
                del self._tools[spec.name]

        return dispose

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(sorted(self._tools.values(), key=lambda spec: spec.name))

    def schemas(self) -> list[dict[str, Any]]:
        """Every tool as the model sees it."""
        return [spec.schema() for spec in self]

    def describe(self) -> list[dict[str, Any]]:
        """Every tool as an operator sees it, host fields included."""
        return [
            {
                **spec.schema(),
                "dangerous": spec.dangerous,
                "timeout_s": spec.timeout_s,
                "source": spec.source,
            }
            for spec in self
        ]

    # -- execution ---------------------------------------------------------

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str = "",
        owner: str = "default",
    ) -> ToolResult:
        """Run one tool call. Never raises; failures come back as results."""
        args = dict(arguments or {})
        run = ToolRun(tool=name, call_id=call_id or f"call-{time.monotonic_ns():x}", owner=owner)
        run.arguments = args

        spec = self._tools.get(name)
        if spec is None:
            # Naming the alternatives matters: the usual cause is a model
            # inventing a plausible tool, and a list is what corrects it.
            known = ", ".join(sorted(self._tools)) or "none"
            return self._fail(run, ToolError.NO_TOOL, f"no tool named {name!r} (have: {known})")

        started = time.perf_counter()

        denial = await self._pre_execute(spec, run)
        if denial is not None:
            return self._fail(run, denial.code, denial.reason, started)

        try:
            value = await self._execute(spec, run, args)
        except TimeoutError:
            return self._fail(run, ToolError.TIMEOUT, f"{name} exceeded {spec.timeout_s}s", started)
        except asyncio.CancelledError:
            return self._fail(run, ToolError.CANCELLED, "cancelled", started)
        except ToolInvocationError as exc:
            # The tool rejected its own arguments. That is a normal outcome
            # worth distinguishing from a crash, because the model can fix it.
            return self._fail(run, exc.code, str(exc), started)
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return self._fail(run, ToolError.FAILED, f"{type(exc).__name__}: {exc}", started)

        content, truncated = _render(value)
        result = ToolResult(
            tool=name,
            call_id=run.call_id,
            ok=True,
            content=content,
            value=value,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            truncated=truncated,
        )
        await self._post_execute(spec, run, result)
        return result

    async def _pre_execute(self, spec: ToolSpec, run: ToolRun) -> Denial | None:
        """Ask the policy layer whether this call may proceed."""
        if self._ctx is None:
            return None
        for outcome in await self._ctx.emit("tools/pre-execute", spec, run):
            if isinstance(outcome, Denial):
                return outcome
        return None

    async def _execute(self, spec: ToolSpec, run: ToolRun, args: dict[str, Any]) -> Any:
        body = spec.execute(args, run)
        if not inspect.isawaitable(body):
            # A synchronous tool is legal and common (a pure computation over
            # already-loaded state); only the await is conditional.
            return body
        if spec.timeout_s is None:
            return await body
        return await asyncio.wait_for(body, timeout=spec.timeout_s)

    async def _post_execute(self, spec: ToolSpec, run: ToolRun, result: ToolResult) -> None:
        if self._ctx is None:
            return
        await self._ctx.emit("tools/post-execute", spec, run, result)

    def _fail(
        self,
        run: ToolRun,
        code: ToolError,
        detail: str,
        started: float | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool=run.tool,
            call_id=run.call_id,
            ok=False,
            # The model reads `content`, so the failure has to be legible
            # there too — an empty content with an error code beside it is a
            # result the model cannot act on.
            content=f"error ({code.value}): {detail}",
            error=code,
            detail=detail,
            duration_ms=((time.perf_counter() - started) * 1000.0) if started else 0.0,
        )


class ToolInvocationError(Exception):
    """Raised by a tool body to reject its own arguments.

    Distinct from an unexpected exception: this one says "the request was
    wrong", which the model can act on, rather than "the tool broke".
    """

    def __init__(self, message: str, code: ToolError = ToolError.BAD_ARGUMENTS) -> None:
        super().__init__(message)
        self.code = code


def _render(value: Any) -> tuple[str, bool]:
    """Project a tool's return value into model-facing text."""
    if value is None:
        return "", False
    text = value if isinstance(value, str) else _as_json(value)
    if len(text) <= MAX_RESULT_CHARS:
        return text, False
    # Say what was cut. A silently truncated result is one the model reasons
    # about as if it were complete.
    head = text[:MAX_RESULT_CHARS]
    dropped = len(text) - MAX_RESULT_CHARS
    return f"{head}\n\n[truncated: {dropped:,} more characters]", True


def _as_json(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return repr(value)
