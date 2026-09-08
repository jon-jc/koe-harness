"""HTTP and WebSocket surface.

The realtime endpoint is a WebSocket rather than a series of HTTP posts because
the transport has to carry two different things in two directions at once:
audio frames up, and partial hypotheses down, continuously, for minutes at a
time. Chunked POSTs give you the first half and nothing for the second.

**Control frames are JSON, audio frames are binary.** Base64-ing PCM into JSON
would inflate every frame by a third for no benefit, on the one path where
bandwidth is continuous rather than occasional.

**Every session is scoped.** A connection gets a child of the root kernel
context, so its subscriptions, tasks and buffers are owned by that scope and
released when the socket closes -- including when it closes by the browser tab
disappearing mid-utterance, which is the normal way voice sessions end.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from koe import __version__
from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Segment, Transcript, attribute_speakers
from koe.kernel.context import Context
from koe.minutes.generator import MinutesGenerator
from koe.minutes.schema import Minutes
from koe.pipeline.session import SessionConfig, StreamingSession
from koe.providers.mock import MEETING_JA, MockASR, MockDiarization, MockLLM
from koe.routing.budget import Budget, Priority
from koe.routing.router import Router
from koe.text.script import Language
from koe.text.tokenize import mecab_available

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 1 << 20  # 1 MiB; a 20ms frame is 640 bytes
MAX_SESSION_SECONDS = 60 * 60 * 4


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


@dataclass
class Services:
    """Everything the API needs, assembled once at startup.

    Held on the app rather than imported at module scope so tests can build a
    server with different backends, and so the process starts without
    credentials.
    """

    ctx: Context
    asr_router: Router[Any]
    llm: Any
    diarizer: Any

    @classmethod
    def default(cls) -> Services:
        """Mock-backed services, so the server runs on a clean checkout."""
        ctx = Context(name="api")
        fast = MockASR(
            name="mock-fast", degradation=0.10, latency_ms=0.0, cost_per_audio_minute_usd=0.002
        )
        accurate = MockASR(
            name="mock-accurate", degradation=0.02, latency_ms=0.0, cost_per_audio_minute_usd=0.010
        )
        router: Router[Any] = Router([fast, accurate])
        ctx.provide("asr", router)
        return cls(
            ctx=ctx,
            asr_router=router,
            llm=MockLLM(default_response=Minutes().model_dump_json()),
            diarizer=MockDiarization(script=MEETING_JA),
        )


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    japanese_tokenizer: str
    providers: list[dict[str, object]]


class TranscribeRequest(BaseModel):
    """Transcribe a scripted meeting -- the credential-free demo path."""

    meeting: str = Field(default="quarterly-ja")
    # Typed as the enum so an unknown value is a 422 at the boundary rather
    # than a ValueError escaping the handler as a 500.
    priority: Priority = Priority.BALANCED
    language: Language = Language.UNKNOWN


class TranscriptResponse(BaseModel):
    text: str
    language: Language
    provider: str
    segments: list[Segment]
    routed_to: str = ""
    fell_back: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class MinutesRequest(BaseModel):
    transcript: Transcript
    language: Language = Language.UNKNOWN


class MinutesResponse(BaseModel):
    minutes: Minutes
    rendered: str
    grounded: int
    total_claims: int
    dropped: list[str]
    repairs: int
    cost_usd: float


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


def build_router(services: Services) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            japanese_tokenizer="mecab" if mecab_available() else "character",
            providers=services.asr_router.health(),
        )

    @router.get("/v1/providers")
    async def providers() -> JSONResponse:
        """Routing health -- which backends are eligible and why."""
        decision = services.asr_router.select(Budget())
        return JSONResponse(
            {
                "ranked": [
                    {
                        "name": c.name,
                        "score": round(c.score, 4),
                        "cost_per_audio_minute_usd": c.info.cost_per_audio_minute_usd,
                        "expected_error_rate": c.info.error_rate_for(Language.JA),
                        "measured": c.info.measured,
                    }
                    for c in decision.ranked
                ],
                "rejected": decision.rejected,
                "breakers": services.asr_router.health(),
            }
        )

    @router.post("/v1/transcribe", response_model=TranscriptResponse)
    async def transcribe(request: TranscribeRequest) -> TranscriptResponse:
        from koe.evaluation.corpus import ALL_SCRIPTS

        script = ALL_SCRIPTS.get(request.meeting, MEETING_JA)
        seconds = max(u.end for u in script)
        audio = AudioChunk(data=bytes(STANDARD_FORMAT.bytes_for(seconds)))

        budget = Budget(priority=request.priority, language=request.language)

        async def call(provider: Any) -> Transcript:
            scoped = MockASR(
                script=script,
                degradation=provider.degradation,
                name=provider.info.name,
                cost_per_audio_minute_usd=provider.info.cost_per_audio_minute_usd,
            )
            return await scoped.transcribe(audio, language=request.language or None)

        started = time.perf_counter()
        result = await services.asr_router.execute(budget, call)
        diarization = await MockDiarization(script=script).diarize(audio)
        fused = attribute_speakers(result.value, diarization)

        return TranscriptResponse(
            text=fused.text,
            language=fused.dominant_language(),
            provider=result.provider,
            segments=fused.segments,
            routed_to=result.provider,
            fell_back=result.fell_back,
            cost_usd=result.decision.ranked[0].info.estimate_audio_cost(seconds),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @router.post("/v1/minutes", response_model=MinutesResponse)
    async def minutes(request: MinutesRequest) -> MinutesResponse:
        generator = MinutesGenerator(services.llm)
        result = await generator.generate(request.transcript, language=request.language or None)
        return MinutesResponse(
            minutes=result.minutes,
            rendered=result.minutes.render(),
            grounded=result.report.grounded,
            total_claims=result.report.total,
            dropped=result.dropped,
            repairs=result.repairs,
            cost_usd=result.usage.cost_usd,
        )

    return router


# --------------------------------------------------------------------------
# websocket
# --------------------------------------------------------------------------


async def _stream_endpoint(websocket: WebSocket, services: Services) -> None:
    """Drive one realtime session over a socket."""
    await websocket.accept()

    # Session-scoped context: everything registered here dies with the socket,
    # including when the browser tab vanishes mid-utterance.
    scope = services.ctx.scope.fork("ws")
    ctx = services.ctx._derive(scope=scope)

    session: StreamingSession | None = None
    outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

    def enqueue(payload: dict[str, Any]) -> None:
        # Never block the audio path on a slow client. A viewer that cannot
        # keep up loses interim frames, which are cosmetic; audio intake and
        # final segments continue regardless.
        with contextlib.suppress(asyncio.QueueFull):
            outbound.put_nowait(payload)

    ctx.on(
        "asr.partial",
        lambda event: enqueue(
            {
                "type": "partial",
                "committed": event.committed,
                "pending": event.pending,
                "language": event.language.value,
            }
        ),
    )
    ctx.on(
        "asr.final",
        lambda segment: enqueue(
            {
                "type": "final",
                "text": segment.text,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "language": segment.language.value,
                "speaker": segment.speaker or "",
            }
        ),
    )
    ctx.on(
        "speech.start",
        lambda event: enqueue({"type": "speech", "state": "start", "at": round(event.at, 2)}),
    )
    ctx.on(
        "speech.end",
        lambda event: enqueue({"type": "speech", "state": "end", "at": round(event.at, 2)}),
    )

    async def pump() -> None:
        while True:
            payload = await outbound.get()
            await websocket.send_json(payload)

    pump_task = asyncio.create_task(pump(), name="koe.ws.pump")

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (text := message.get("text")) is not None:
                import json

                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    enqueue({"type": "error", "message": "control frame is not valid JSON"})
                    continue

                action = control.get("type")
                if action == "start":
                    language = Language(control.get("language", "unknown"))
                    provider = services.asr_router.select(
                        Budget.realtime(language)
                        if control.get("realtime", True)
                        else Budget(language=language)
                    ).chosen
                    session = StreamingSession(
                        ctx,
                        provider,
                        config=SessionConfig(
                            language=language,
                            partial_interval_ms=float(control.get("partial_interval_ms", 500)),
                        ),
                    )
                    await session.start()
                    enqueue(
                        {
                            "type": "started",
                            "session_id": session.session_id,
                            "provider": provider.info.name,
                            "language": language.value,
                        }
                    )
                elif action == "stop":
                    if session is not None:
                        transcript = await session.finish()
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                "text": transcript.text,
                                "duration": round(transcript.duration, 2),
                                "cost_usd": round(session.usage.cost_usd, 6),
                                "segments": [
                                    {
                                        "text": s.text,
                                        "start": round(s.start, 2),
                                        "end": round(s.end, 2),
                                        "speaker": s.speaker or "",
                                    }
                                    for s in transcript.final_segments
                                ],
                            }
                        )
                        session = None
                    break
                continue

            if (payload := message.get("bytes")) is not None:
                if session is None:
                    enqueue({"type": "error", "message": "send a start frame first"})
                    continue
                if len(payload) > MAX_FRAME_BYTES:
                    enqueue({"type": "error", "message": "audio frame too large"})
                    continue
                if session.duration > MAX_SESSION_SECONDS:
                    enqueue({"type": "error", "message": "session length limit reached"})
                    break
                await session.push_audio(payload)

    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("streaming session failed")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": "internal error"})
    finally:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump_task
        if session is not None:
            with contextlib.suppress(Exception):
                await session.finish()
        scope.dispose()
        with contextlib.suppress(Exception):
            await websocket.close()


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def _web_root() -> Path:
    """Where the built web client lives, relative to the installed package."""
    return Path(__file__).resolve().parents[3] / "web"


def create_app(services: Services | None = None) -> FastAPI:
    """Build the ASGI application."""
    resolved = services or Services.default()

    app = FastAPI(
        title="koe",
        version=__version__,
        description="声 — bilingual JA/EN voice AI harness",
    )
    app.state.services = resolved
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_router(resolved))

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await _stream_endpoint(websocket, resolved)

    # The built client, when present. Mounted rather than read per request so
    # the bundle is served with proper caching headers.
    web_root = _web_root()
    if (web_root / "dist").is_dir():
        app.mount("/static", StaticFiles(directory=web_root / "dist"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        client = web_root / "index.html"
        if client.exists():
            return HTMLResponse(client.read_text(encoding="utf-8"))
        # The API is useful without the client; say so rather than 404-ing,
        # since this is also what a health-checking human hits first.
        return HTMLResponse(
            "<h1>koe</h1><p>API is running. Build the client with "
            "<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>.</p>",
        )

    return app


app = create_app()
