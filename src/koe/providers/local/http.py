"""A small JSON-over-HTTP client, on the standard library.

koe reaches local model servers with :mod:`urllib.request` rather than httpx or
a vendor SDK, and the reason is the whole point of this package: **local models
must work on a bare install.** ``pip install koe-harness[api]`` and an Ollama
already running on the machine should be a working setup, with no key, no
account, and no second install step. Requiring the ``openai`` package to reach
``127.0.0.1`` would make the offline path depend on the online one.

The cost is that urllib is synchronous, so every call goes through
:func:`asyncio.to_thread`. That is the correct trade here — these are requests
to loopback, the thread is parked on a socket, and the alternative is a
dependency on the event-loop-native HTTP client of the month.

**Timeouts are separate for connect and read, and this matters more locally
than remotely.** A local server that is not running refuses the connection
instantly, which is what makes probing five ports cheap. A local server that
*is* running may take thirty seconds to answer the first request while it loads
a seven-billion-parameter model from disk into RAM. One timeout covering both
either gives up on a loading model or spends thirty seconds discovering that
nothing is listening.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

#: Long enough to cross loopback and be refused, short enough that probing
#: every well-known port costs nothing perceptible.
PROBE_TIMEOUT_S = 1.5

#: A first request to a local server can include loading the weights.
GENERATE_TIMEOUT_S = 300.0


# `timeout` below is a *socket* timeout, and ruff's ASYNC109 asks for
# `asyncio.timeout` instead. Not here: these calls block inside urllib on a
# worker thread, and `asyncio.to_thread` cannot be cancelled. An asyncio
# timeout would return control to the caller while leaving the thread parked on
# the socket until the OS gave up — a leak on exactly the path that exists to
# survive an unresponsive local server. The socket timeout is the only one that
# actually ends the work.


class LocalHTTPError(Exception):
    """A request to a local server failed.

    Carries `status` when the server answered and refused, and leaves it None
    when nothing answered at all -- the difference between "LM Studio is
    running but has no model loaded" and "LM Studio is not running", which are
    different things to tell a user.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def unreachable(self) -> bool:
        return self.status is None


def _request(
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
    api_key: str = "",
) -> Any:
    """One blocking JSON request. Runs on a worker thread."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        # Some local servers are configured to require one, and vLLM behind a
        # reverse proxy commonly is. Sent when present, never required.
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            # Best-effort: a server that errored may also fail to describe it,
            # and the status code alone is still worth reporting.
            detail = exc.read().decode("utf-8", "replace")[:400]
        raise LocalHTTPError(
            f"{exc.code} from {url}{f': {detail}' if detail else ''}", status=exc.code
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LocalHTTPError(f"could not reach {url}: {exc}") from exc

    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        # A local server answering HTML on the API path is almost always its
        # web UI, which means the base URL is the origin rather than the API.
        raise LocalHTTPError(f"{url} did not return JSON: {exc}") from exc


async def get_json(
    url: str,
    *,
    timeout: float = PROBE_TIMEOUT_S,  # noqa: ASYNC109 - socket timeout; see the note above
    api_key: str = "",
) -> Any:
    return await asyncio.to_thread(_request, url, payload=None, timeout=timeout, api_key=api_key)


async def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = GENERATE_TIMEOUT_S,  # noqa: ASYNC109 - socket timeout; see the note above
    api_key: str = "",
) -> Any:
    return await asyncio.to_thread(_request, url, payload=payload, timeout=timeout, api_key=api_key)
