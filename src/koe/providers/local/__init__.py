"""Inference on the user's own machine.

The claim koe makes elsewhere is that it is a harness: backends are
interchangeable and the router picks between them. Local models are the test of
that claim, because they are the case where every assumption the hosted
providers share stops holding. There is no key. There is no vendor SDK. Cost is
zero, latency is an order of magnitude worse, and the model is whatever the user
happened to pull last week.

Four positions, all of them consequences of that.

**Local models must work on a bare install.** ``pip install koe-harness[api]``
with an Ollama already running is a complete setup. So :mod:`.http` is
:mod:`urllib` and nothing else — making the offline path depend on the
``openai`` package would be an odd thing to require of someone whose reason for
running locally is that they do not want an account.

**The port is a probe, not a setting.** Someone who installed a GUI to avoid
knowing which port it binds should not be asked for it. :mod:`.discovery`
sweeps the well-known ones concurrently; on loopback a refused connection
returns immediately, so the common case costs nothing.

**Zero cost is not an advantage on its own, and the metadata says so.** A
provider reporting no cost and nothing else would win every routing decision,
and koe would feel broken rather than free. The local backends declare
pessimistic latency and honest per-language quality priors, which is what lets
the router make the trade it exists to make.

**Local recognition is the privacy claim, and it is scoped.** With
:mod:`.whisper` running, audio never leaves the device — which is what makes koe
usable on a confidential meeting at all. koe only says so for loopback; a model
server elsewhere on the network is a fine deployment and a different promise.
"""

from koe.providers.local.discovery import Discovered, Discovery, LocalModel, discover
from koe.providers.local.endpoints import (
    SERVERS_BY_ID,
    WELL_KNOWN,
    LocalServer,
    is_loopback,
    normalize_base_url,
    openai_base_candidates,
)
from koe.providers.local.http import LocalHTTPError
from koe.providers.local.llm import LocalLLM
from koe.providers.local.resolve import Choice, resolve_asr, resolve_llm
from koe.providers.local.whisper import (
    DEFAULT_SIZE,
    SIZES,
    SIZES_BY_ID,
    LocalWhisperASR,
    WhisperSize,
    usable_for,
)

__all__ = [
    "DEFAULT_SIZE",
    "SERVERS_BY_ID",
    "SIZES",
    "SIZES_BY_ID",
    "WELL_KNOWN",
    "Choice",
    "Discovered",
    "Discovery",
    "LocalHTTPError",
    "LocalLLM",
    "LocalModel",
    "LocalServer",
    "LocalWhisperASR",
    "WhisperSize",
    "discover",
    "is_loopback",
    "normalize_base_url",
    "openai_base_candidates",
    "resolve_asr",
    "resolve_llm",
    "usable_for",
]
