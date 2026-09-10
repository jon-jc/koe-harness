"""Finding the model servers already running on this machine.

Someone who has installed Ollama and pulled a model has done the hard part.
Asking them next for a base URL and a model identifier is asking them to
re-enter information the machine already has, and it is the step at which
"local models are supported" quietly becomes "local models are possible".

So koe probes. Every well-known port, concurrently, with a short timeout, and
what comes back is a list of servers and the models each one is holding.

**Probing concurrently is what makes it free.** Five ports at a 1.5 s timeout is
1.5 s if they are all dead and run in parallel, and 7.5 s if they are not. A
refused connection on loopback returns in microseconds, so the common case --
nothing running -- costs nothing at all.

**A server with no models is reported, not hidden.** LM Studio running with
nothing loaded is a different problem from LM Studio not running, and it has a
different fix. Collapsing both into "no local models found" sends someone to
reinstall software that is already working.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from koe.providers.local.endpoints import (
    WELL_KNOWN,
    LocalServer,
    is_loopback,
    normalize_base_url,
    openai_base_candidates,
)
from koe.providers.local.http import PROBE_TIMEOUT_S, LocalHTTPError, get_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalModel:
    """One model a local server is offering."""

    id: str
    #: Parameter count and quantization where the server reports them. Ollama
    #: does; the OpenAI-compatible listing does not. This is the number that
    #: decides whether a model fits in RAM, so it is worth the extra call.
    parameters: str = ""
    quantization: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parameters": self.parameters,
            "quantization": self.quantization,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class Discovered:
    """A local server that answered, and what it is holding."""

    server_id: str
    label: str
    base_url: str
    models: tuple[LocalModel, ...] = ()
    #: Set when the server answered but the listing failed, so the UI can say
    #: "running, but could not list models" rather than inventing a reason.
    note: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "label": self.label,
            "base_url": self.base_url,
            "ready": self.ready,
            "local": is_loopback(self.base_url),
            "note": self.note,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True, slots=True)
class Discovery:
    """The result of one sweep."""

    servers: tuple[Discovered, ...] = ()
    #: Every well-known server that did not answer, so the UI can offer an
    #: install link rather than a blank panel.
    absent: tuple[LocalServer, ...] = field(default_factory=tuple)

    @property
    def any_models(self) -> bool:
        return any(server.ready for server in self.servers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "servers": [server.to_dict() for server in self.servers],
            "absent": [
                {"server_id": s.id, "label": s.label, "port": s.port, "docs_url": s.docs_url}
                for s in self.absent
            ],
            "any_models": self.any_models,
        }


def _models_from_openai(payload: Any) -> tuple[LocalModel, ...]:
    """Parse ``GET /v1/models``."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return ()
    models = []
    for entry in data:
        identifier = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(identifier, str) and identifier:
            models.append(LocalModel(id=identifier))
    return tuple(models)


def _models_from_ollama(payload: Any) -> tuple[LocalModel, ...]:
    """Parse Ollama's ``GET /api/tags``, which says more than /v1/models."""
    data = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return ()

    models = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            continue
        raw_details = entry.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        models.append(
            LocalModel(
                id=name,
                parameters=str(details.get("parameter_size") or ""),
                quantization=str(details.get("quantization_level") or ""),
                size_bytes=int(entry.get("size") or 0),
            )
        )
    return tuple(models)


async def list_models(
    base_url: str,
    *,
    api_key: str = "",
    timeout: float = PROBE_TIMEOUT_S,  # noqa: ASYNC109 - socket timeout; see the note above
) -> tuple[LocalModel, ...]:
    """Models offered at `base_url`, trying each plausible API path."""
    for candidate in openai_base_candidates(base_url):
        try:
            payload = await get_json(f"{candidate}/models", timeout=timeout, api_key=api_key)
        except LocalHTTPError:
            continue
        models = _models_from_openai(payload)
        if models:
            return models
    return ()


async def probe(
    server: LocalServer,
    *,
    timeout: float = PROBE_TIMEOUT_S,  # noqa: ASYNC109 - socket timeout; see the note above
) -> Discovered | None:
    """Ask one well-known server what it has. None means nothing answered."""
    origin = f"http://127.0.0.1:{server.port}"

    models: tuple[LocalModel, ...] = ()
    answered = False

    if server.native_models_path:
        # Preferred where it exists: the native listing carries the parameter
        # count and quantization, and the OpenAI-compatible one does not.
        try:
            payload = await get_json(f"{origin}{server.native_models_path}", timeout=timeout)
        except LocalHTTPError as exc:
            if not exc.unreachable:
                answered = True
        else:
            answered = True
            models = _models_from_ollama(payload)

    if not models:
        found = await list_models(server.base_url, timeout=timeout)
        if found:
            answered = True
            models = found
        elif not answered:
            # Nothing from the native path either: settle whether anything is
            # listening at all, so a running-but-empty server is distinguished
            # from an absent one.
            try:
                await get_json(f"{server.base_url}/models", timeout=timeout)
            except LocalHTTPError as exc:
                if exc.unreachable:
                    return None
            answered = True

    if not answered:
        return None

    return Discovered(
        server_id=server.id,
        label=server.label,
        base_url=server.base_url,
        models=models,
        note="" if models else "running, but no models are loaded",
    )


async def discover(
    *,
    servers: tuple[LocalServer, ...] = WELL_KNOWN,
    timeout: float = PROBE_TIMEOUT_S,  # noqa: ASYNC109 - socket timeout; see the note above
) -> Discovery:
    """Sweep every well-known port at once."""
    results = await asyncio.gather(
        *(probe(server, timeout=timeout) for server in servers),
        return_exceptions=True,
    )

    found: list[Discovered] = []
    missing: list[LocalServer] = []
    for server, result in zip(servers, results, strict=True):
        if isinstance(result, BaseException):
            # A probe is best-effort by definition. One server misbehaving must
            # not cost the user the others.
            logger.debug("probe of %s failed: %s", server.id, result)
            missing.append(server)
        elif result is None:
            missing.append(server)
        else:
            found.append(result)

    return Discovery(servers=tuple(found), absent=tuple(missing))


async def describe(base_url: str, *, api_key: str = "") -> Discovered | None:
    """Probe one base URL the user supplied, rather than a well-known port."""
    base = normalize_base_url(base_url)
    if not base:
        return None
    models = await list_models(base, api_key=api_key)
    if not models:
        try:
            await get_json(f"{openai_base_candidates(base)[-1]}/models", api_key=api_key)
        except LocalHTTPError as exc:
            if exc.unreachable:
                return None
    # A URL the user typed still gets the friendly name when koe recognizes the
    # port, because "Ollama · llama3.2:3b" is what they are looking for in the
    # panel and "http://127.0.0.1:11434/v1 · llama3.2:3b" is the same fact
    # spelled less usefully. Selecting a model from a discovered server writes
    # a base URL, so without this the act of choosing degrades the label.
    known = next(
        (s for s in WELL_KNOWN if s.base_url == base or f"http://127.0.0.1:{s.port}" == base), None
    )
    return Discovered(
        server_id=known.id if known else "custom",
        label=known.label if known else base,
        base_url=base,
        models=models,
        note="" if models else "reachable, but no models are loaded",
    )
