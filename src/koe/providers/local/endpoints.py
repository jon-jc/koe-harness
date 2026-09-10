"""Where a local model server lives, and how to talk to it.

Local inference has converged on one interface. Ollama, LM Studio, llama.cpp's
server, vLLM, Jan and LocalAI all expose OpenAI's ``/v1/chat/completions``,
whatever else they also offer. So koe needs exactly one client, and the only
real questions are *which port* and *which path*.

**The port is a probe, not a setting.** Asking someone which port Ollama uses is
asking them to know something they installed a GUI specifically to avoid
knowing. The ports below are the defaults those projects ship, and koe tries
them.

**The path needs normalizing, and this is where naive clients break.** People
type ``localhost:11434`` because that is what the Ollama docs show, and the
OpenAI-compatible API is at ``localhost:11434/v1``. People also paste
``http://localhost:1234/v1`` because that is what LM Studio's own UI shows. Both
have to work, so a bare origin gets ``/v1`` appended and one that already ends
in ``/v1`` is left alone. LM Studio's native REST base (``/api/v0`` or
``/api/v1``) has its OpenAI-compatible sibling at ``/v1``, so that is rewritten
rather than rejected. The candidate-list approach is OpenWhispr's; the reason it
exists is that every one of these is something a real user actually pastes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class LocalServer:
    """A local inference server koe knows how to look for."""

    id: str
    label: str
    port: int
    #: What to append to the origin to reach the OpenAI-compatible API.
    path: str = "/v1"
    #: Ollama's own model listing is richer than /v1/models -- it carries the
    #: parameter count and quantization, which is what tells someone whether a
    #: model will fit in their RAM. Empty where there is no native listing.
    native_models_path: str = ""
    docs_url: str = ""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.path}"


#: The defaults these projects ship. Order is the order koe probes in, and it
#: is roughly by how likely someone running local models is to have it.
WELL_KNOWN: tuple[LocalServer, ...] = (
    LocalServer(
        id="ollama",
        label="Ollama",
        port=11434,
        native_models_path="/api/tags",
        docs_url="https://ollama.com/download",
    ),
    LocalServer(
        id="lmstudio",
        label="LM Studio",
        port=1234,
        docs_url="https://lmstudio.ai",
    ),
    LocalServer(
        id="llamacpp",
        label="llama.cpp server",
        port=8080,
        docs_url="https://github.com/ggml-org/llama.cpp",
    ),
    LocalServer(
        id="vllm",
        label="vLLM",
        port=8000,
        docs_url="https://docs.vllm.ai",
    ),
    LocalServer(
        id="jan",
        label="Jan",
        port=1337,
        docs_url="https://jan.ai",
    ),
)

SERVERS_BY_ID = {server.id: server for server in WELL_KNOWN}

#: LM Studio's native REST base. Its OpenAI-compatible sibling is at /v1.
_NATIVE_API_SUFFIX = re.compile(r"^(?P<origin>.+?)/api/v[01]$", re.IGNORECASE)


def normalize_base_url(text: str) -> str:
    """Clean up a base URL a person typed. Empty means unusable.

    A scheme is added when missing, because ``localhost:11434`` is what the
    docs of every one of these projects shows and it is not a valid URL.
    """
    candidate = (text or "").strip()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"http://{candidate}"

    parts = urlsplit(candidate)
    if not parts.hostname:
        return ""
    # Query and fragment are meaningless on an API base and are usually a
    # paste accident; dropping them is kinder than failing on them.
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def openai_base_candidates(text: str) -> tuple[str, ...]:
    """Bases to try, in order, for an OpenAI-compatible API.

    Returns both the URL as given and the ``/v1`` form when they differ, so a
    bare origin and a fully-specified base both work without the user having to
    know which one this build wanted.
    """
    base = normalize_base_url(text)
    if not base:
        return ()

    native = _NATIVE_API_SUFFIX.match(base)
    if native:
        return (base, f"{native.group('origin')}/v1")
    if base.endswith("/v1"):
        return (base,)
    return (base, f"{base}/v1")


def is_loopback(text: str) -> bool:
    """Whether this URL points at the machine koe is running on.

    Used to decide whether "your audio never leaves this device" is a claim koe
    is entitled to make. A user is free to point koe at a model server on their
    own network, and that is a perfectly good deployment -- but it is not the
    same promise, and the UI must not make the stronger one on its behalf.
    """
    base = normalize_base_url(text)
    if not base:
        return False
    host = (urlsplit(base).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost")
