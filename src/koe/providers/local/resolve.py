"""Choosing a local backend, and saying why.

Selection is separated from the providers themselves because the interesting
part is not "construct a LocalLLM" -- it is the decision, and the decision has
to be explicable. A user who expected their local model and got the mock needs
to know which step declined, and "no local models found" is not that.

So every function here returns a :class:`Choice`: what was picked, or nothing,
and a sentence saying why. The API hands that sentence to the settings panel
verbatim.

**A configured key wins over a discovered server by default.** Someone who
pasted an API key expressed a preference; a server that happens to be listening
on 11434 did not. ``prefer_local_llm`` inverts that for people who want local
first, which is the setting a privacy-motivated user is looking for and should
not have to find by deleting their key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from koe.providers.local.discovery import Discovered, describe, discover
from koe.providers.local.llm import LocalLLM
from koe.providers.local.whisper import DEFAULT_SIZE, SIZES_BY_ID, LocalWhisperASR
from koe.providers.local.whisper import available as whisper_available

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Choice:
    """What was selected, and the sentence explaining it."""

    provider: Any = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.provider is not None


async def resolve_llm(settings: Any) -> Choice:
    """A local LLM from configuration, or from whatever is running.

    An explicit `local_llm_base_url` is honoured as given -- including when it
    is unreachable, which is reported rather than silently swapped for a
    discovered server. Someone who typed an address wants that address, and a
    fallback that quietly picks something else makes a typo undebuggable.
    """
    configured = (getattr(settings, "local_llm_base_url", "") or "").strip()
    api_key = (getattr(settings, "local_llm_api_key", "") or "").strip()
    wanted = (getattr(settings, "local_llm_model", "") or "").strip()

    if configured:
        found = await describe(configured, api_key=api_key)
        if found is None:
            return Choice(reason=f"nothing answered at {configured}")
        return _from_discovered(found, wanted, api_key)

    if not getattr(settings, "discover_local_models", True):
        return Choice(reason="local discovery is off")

    sweep = await discover()
    ready = [server for server in sweep.servers if server.ready]
    if not ready:
        running = [server.label for server in sweep.servers]
        if running:
            return Choice(
                reason=f"{', '.join(running)} is running but has no model loaded",
            )
        return Choice(reason="no local model server is running")

    return _from_discovered(ready[0], wanted, api_key)


def _from_discovered(found: Discovered, wanted: str, api_key: str) -> Choice:
    if not found.models:
        return Choice(reason=found.note or f"{found.label} has no model loaded")

    model = found.models[0].id
    if wanted:
        # Exact first, then prefix: people write "qwen2.5" for a model Ollama
        # calls "qwen2.5:7b", and refusing that is pedantry.
        exact = next((m.id for m in found.models if m.id == wanted), "")
        prefix = next((m.id for m in found.models if m.id.startswith(wanted)), "")
        if not exact and not prefix:
            offered = ", ".join(m.id for m in found.models[:6])
            return Choice(reason=f"{found.label} does not have {wanted!r} — it has {offered}")
        model = exact or prefix

    return Choice(
        provider=LocalLLM(
            model=model,
            base_url=found.base_url,
            api_key=api_key,
            label=found.server_id,
        ),
        reason=f"{found.label} · {model}",
    )


def resolve_asr(settings: Any) -> Choice:
    """Local Whisper, if it is switched on and installable.

    Synchronous: unlike the LLM path there is nothing to probe. Either the
    library is importable or it is not, and the model file is faster-whisper's
    problem to fetch on first use.
    """
    if not getattr(settings, "local_asr_enabled", False):
        return Choice(reason="local recognition is off")

    if not whisper_available():
        return Choice(
            reason="local recognition needs the 'asr' extra: pip install 'koe-harness[asr]'"
        )

    size = (getattr(settings, "local_asr_model", "") or DEFAULT_SIZE).strip()
    if size not in SIZES_BY_ID:
        return Choice(reason=f"unknown Whisper size {size!r}")

    provider = LocalWhisperASR(
        size=size,
        device=(getattr(settings, "local_asr_device", "") or "auto"),
        compute_type=(getattr(settings, "local_asr_compute_type", "") or "default"),
    )
    return Choice(provider=provider, reason=f"local Whisper · {size}")
