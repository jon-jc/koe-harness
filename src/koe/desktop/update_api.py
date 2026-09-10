"""The updater's HTTP surface, mounted by the desktop app and nothing else.

The web client is how a person sees and drives updates — the status in
Settings → About, "Restart to update" in the sidebar — and the client only
speaks HTTP to the embedded server. So the desktop process adds these routes
to the app it serves; a server deployment never has them, and the client hides
the controls when ``GET /v1/update`` is not there.

**Changes refuse cross-origin requests.** The server listens on loopback, but
any page open in the person's real browser can still send requests to
``127.0.0.1``. A request carrying an ``Origin`` that is not this server's own is
refused, so a web page cannot start an install. The worst it could otherwise
do is install an already-verified koe release — but "a website can restart
your app" is not a property worth having at any severity.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from koe.desktop.updater import UpdateError, Updater

#: Long enough for the response to leave the socket before the process exits.
EXIT_DELAY_S = 0.6


class AutoBody(BaseModel):
    enabled: bool


class ApplyBody(BaseModel):
    relaunch: bool = True


def require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        # Not a browser request made on a page's behalf.
        return
    own = f"{request.url.scheme}://{request.url.netloc}"
    if origin.rstrip("/") != own:
        raise HTTPException(status_code=403, detail="cross-origin update requests are refused")


def build_update_router(
    updater: Updater,
    *,
    on_apply: Callable[[], None],
    on_auto_changed: Callable[[bool], None] = lambda enabled: None,
) -> APIRouter:
    router = APIRouter(prefix="/v1/update", tags=["update"])

    @router.get("")
    async def status() -> dict[str, Any]:
        return updater.snapshot().to_dict()

    @router.post("/check")
    async def check(request: Request) -> dict[str, Any]:
        require_same_origin(request)
        # A worker thread: the check makes network requests and may download
        # ~50 MB, and the event loop is also carrying the recording socket.
        result = await asyncio.to_thread(updater.check)
        return result.to_dict()

    @router.post("/auto")
    async def auto(body: AutoBody, request: Request) -> dict[str, Any]:
        require_same_origin(request)
        updater.set_auto(body.enabled)
        on_auto_changed(body.enabled)
        return updater.snapshot().to_dict()

    @router.post("/apply")
    async def apply(body: ApplyBody, request: Request) -> dict[str, Any]:
        require_same_origin(request)
        try:
            updater.apply(relaunch=body.relaunch)
        except UpdateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"could not start the installer: {exc}"
            ) from exc
        # Exit after the response is on its way. The installer is already
        # running and waiting for this process to end before it touches a file.
        threading.Timer(EXIT_DELAY_S, on_apply).start()
        return updater.snapshot().to_dict()

    return router
