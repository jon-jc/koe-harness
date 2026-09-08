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
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from koe import __version__
from koe.api.demo import drive_demo, retime
from koe.api.middleware import RequestContextMiddleware
from koe.config import Settings, get_settings
from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Segment, Transcript, attribute_speakers
from koe.kernel.context import Context
from koe.minutes.demo import demo_llm
from koe.minutes.generator import MinutesGenerator
from koe.minutes.schema import Minutes
from koe.pipeline.session import SessionConfig, StreamingSession
from koe.providers.credentials import (
    PROVIDERS_BY_ID,
    CredentialError,
    CredentialStore,
    Source,
)
from koe.providers.mock import MEETING_JA, MockASR, MockDiarization
from koe.providers.verify import verify as verify_credential
from koe.routing.budget import Budget, Priority
from koe.routing.router import Router
from koe.telemetry.ledger import BudgetExceeded, CostLedger
from koe.telemetry.metrics import METRICS, Metrics, configure_logging
from koe.text.script import Language
from koe.text.tokenize import mecab_available

logger = logging.getLogger(__name__)


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
    settings: Settings = field(default_factory=get_settings)
    credentials: CredentialStore = field(default_factory=CredentialStore)
    ledger: CostLedger = field(default_factory=CostLedger)
    metrics: Metrics = field(default_factory=lambda: METRICS)
    #: Bounds concurrent streaming sessions. Each holds an audio buffer and a
    #: decoder, so without a cap the failure mode under load is memory
    #: exhaustion rather than a clean rejection.
    _sessions: asyncio.Semaphore = field(init=False, repr=False)
    #: Set whenever no session is active. Shutdown waits on this rather than
    #: polling, so a drain finishes the instant the last caller hangs up
    #: instead of on the next poll tick.
    _idle: asyncio.Event = field(init=False, repr=False)
    active_sessions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._sessions = asyncio.Semaphore(self.settings.max_concurrent_sessions)
        self.refresh_llm()
        self._idle = asyncio.Event()
        self._idle.set()

    def refresh_llm(self) -> str:
        """Point the LLM service at whichever credential is configured.

        Called at startup and again whenever a key is added or removed, so a
        user who pastes a key gets the real model on their next request rather
        than after a restart. Registering through the context rather than only
        assigning the attribute is what lets dependent plugins rebuild against
        the new client — the reason the kernel tracks services reactively.
        """
        chosen = "mock"
        provider: Any = demo_llm()

        if not self.settings.force_mock_providers:
            # A key alone is not enough: the vendor SDK is an optional
            # dependency, and selecting a backend this build cannot import
            # would show "openai" in the UI and then fail on the first
            # request. Better to stay on the mock and say why.
            for candidate in ("anthropic", "openai"):
                spec = PROVIDERS_BY_ID[candidate]
                key = self.credentials.resolve(candidate)
                if not key:
                    continue
                if not spec.available:
                    logger.warning(
                        "credential present but client library missing",
                        extra={"provider": candidate, "client_module": spec.client_module},
                    )
                    continue
                if candidate == "anthropic":
                    from koe.providers.llm.anthropic import AnthropicLLM

                    provider = AnthropicLLM(api_key=key, model=self.settings.llm_model)
                else:
                    from koe.providers.llm.openai import OpenAILLM

                    provider = OpenAILLM(api_key=key)
                chosen = candidate
                break

        self.llm = provider
        self.ctx.provide("llm", provider, replace=True)
        logger.info("llm provider selected", extra={"provider": chosen})
        return chosen

    @contextlib.asynccontextmanager
    async def session_slot(self) -> AsyncIterator[bool]:
        """Reserve a session slot, or yield False if the server is full.

        Rejecting immediately is kinder than queueing: a caller waiting behind
        a full server records audio nobody is transcribing, and would rather be
        told now.
        """
        if self._sessions.locked():
            yield False
            return
        await self._sessions.acquire()
        self.active_sessions += 1
        self._idle.clear()
        try:
            yield True
        finally:
            self.active_sessions -= 1
            self._sessions.release()
            if self.active_sessions == 0:
                self._idle.set()

    async def wait_until_idle(self) -> None:
        """Block until no session is active.

        Deliberately has no timeout parameter: bounding the wait is the
        caller's decision, expressed with ``asyncio.timeout`` at the call site.
        """
        await self._idle.wait()

    @classmethod
    def default(cls, settings: Settings | None = None) -> Services:
        """Mock-backed services, so the server runs on a clean checkout."""
        resolved = settings or get_settings()
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
            # The demo LLM returns a plausible 議事録 containing one
            # deliberately fabricated claim, so the guardrail is visible.
            llm=demo_llm(),
            diarizer=MockDiarization(script=MEETING_JA),
            settings=resolved,
            ledger=CostLedger(
                session_limit_usd=resolved.session_budget_usd,
                tenant_limit_usd=resolved.tenant_budget_usd,
            ),
        )


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    environment: str = "local"
    japanese_tokenizer: str
    providers: list[dict[str, object]]
    active_sessions: int = 0
    max_sessions: int = 0


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


class CredentialRequest(BaseModel):
    """A key being stored.

    `repr=False` on the field keeps the secret out of pydantic's repr, which is
    what ends up in a validation error and therefore in a log line.
    """

    key: str = Field(min_length=8, max_length=512, repr=False)


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
        """Liveness. Cheap and dependency-free, so it answers under load."""
        return HealthResponse(
            environment=services.settings.environment,
            japanese_tokenizer="mecab" if mecab_available() else "character",
            providers=services.asr_router.health(),
            active_sessions=services.active_sessions,
            max_sessions=services.settings.max_concurrent_sessions,
        )

    @router.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness. Distinct from liveness on purpose.

        A saturated server is alive but should stop receiving new sessions, so
        a load balancer needs to be able to ask a different question than a
        restart supervisor does.
        """
        at_capacity = services.active_sessions >= services.settings.max_concurrent_sessions
        healthy_providers = [b for b in services.asr_router.health() if b["state"] != "open"]
        ready = bool(healthy_providers) and not at_capacity
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "at_capacity": at_capacity,
                "healthy_providers": len(healthy_providers),
            },
        )

    @router.get("/metrics")
    async def metrics() -> JSONResponse:
        """Metrics snapshot, plus the cost breakdown.

        Cost sits alongside latency rather than in a separate dashboard,
        because on a multi-model pipeline they are the same decision.
        """
        total = services.ledger.total
        return JSONResponse(
            {
                **services.metrics.snapshot(),
                "cost": {
                    "total_usd": round(total.cost_usd, 6),
                    "audio_seconds": round(total.audio_seconds, 1),
                    "cost_per_audio_hour": round(total.cost_per_audio_hour, 6),
                    "calls": total.calls,
                    "by_model": {
                        name: {
                            "calls": t.calls,
                            "cost_usd": round(t.cost_usd, 6),
                            "cost_per_audio_hour": round(t.cost_per_audio_hour, 6),
                        }
                        for name, t in services.ledger.by_model().items()
                    },
                },
            }
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

    @router.get("/v1/credentials")
    async def list_credentials() -> JSONResponse:
        """Configured providers, with fingerprints rather than keys."""
        return JSONResponse(
            {
                "providers": [info.to_dict() for info in services.credentials.describe_all()],
                "active_llm": getattr(services.llm, "name", "mock"),
                "forced_mock": services.settings.force_mock_providers,
            }
        )

    @router.put("/v1/credentials/{provider}")
    async def set_credential(provider: str, request: CredentialRequest) -> JSONResponse:
        """Store a key and re-point the LLM service at it."""
        spec = PROVIDERS_BY_ID.get(provider)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        if services.credentials.source_of(provider) is Source.ENVIRONMENT:
            # Refusing loudly beats writing a value that will never be read.
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        f"{spec.label} is configured through {spec.env_var}; "
                        "the environment takes precedence over stored keys"
                    ),
                    "code": "env_managed",
                    "context": {"label": spec.label, "env": spec.env_var},
                },
            )
        try:
            info = services.credentials.set(provider, request.key)
        except CredentialError as exc:
            # A code alongside the message, so the bilingual client can say
            # this in the reader's language instead of matching English prose.
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc), "code": exc.code, "context": exc.context},
            )

        active = services.refresh_llm()
        services.metrics.increment("credentials.saved")
        return JSONResponse({"provider": info.to_dict(), "active_llm": active})

    @router.delete("/v1/credentials/{provider}")
    async def delete_credential(provider: str) -> JSONResponse:
        if provider not in PROVIDERS_BY_ID:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        info = services.credentials.delete(provider)
        active = services.refresh_llm()
        return JSONResponse({"provider": info.to_dict(), "active_llm": active})

    @router.post("/v1/credentials/{provider}/verify")
    async def verify_credential_route(provider: str) -> JSONResponse:
        """Check a key against the live API.

        Separate from saving because verification costs a request, and a save
        that silently spends money is a surprise.
        """
        if provider not in PROVIDERS_BY_ID:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        result = await verify_credential(provider, services.credentials)
        info = services.credentials.record_verification(
            provider, result.status, result.detail, result.code
        )
        services.metrics.increment(f"credentials.verify.{result.status.value}")
        return JSONResponse({"provider": info.to_dict()})

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
    async with services.session_slot() as admitted:
        if not admitted:
            # 1013 "try again later" — the close code a client can retry on,
            # unlike a generic failure which looks like a bug to the caller.
            await websocket.accept()
            await websocket.send_json(
                {"type": "error", "message": "server at capacity, retry shortly"}
            )
            await websocket.close(code=1013)
            services.metrics.increment("ws.rejected_at_capacity")
            return
        await _run_session(websocket, services)


async def _run_session(websocket: WebSocket, services: Services) -> None:
    """One admitted realtime session."""
    settings = services.settings
    await websocket.accept()
    services.metrics.increment("ws.sessions")

    # Session-scoped context: everything registered here dies with the socket,
    # including when the browser tab vanishes mid-utterance.
    scope = services.ctx.scope.fork("ws")
    ctx = services.ctx._derive(scope=scope)

    session: StreamingSession | None = None
    demo_task: asyncio.Task[None] | None = None
    last_transcript: Transcript | None = None
    session_started = time.monotonic()
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

    async def run_demo(script: Any) -> None:
        """Play a scripted meeting, then close the session as a client would.

        Nested so it can hand the finished session back to the receive loop.
        A module-level task would leave `session` set here, and teardown would
        then finish it a second time and book its cost twice.
        """
        nonlocal session, last_transcript
        active = session
        if active is None:
            return
        try:
            await drive_demo(
                active,
                script,
                on_level=lambda value: enqueue({"type": "level", "value": round(value, 3)}),
            )
            transcript = await active.finish()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("demo playback failed")
            enqueue({"type": "error", "message": "demo playback failed"})
            session = None
            return

        _record_spend(services, active)
        last_transcript = transcript
        session = None
        enqueue(
            {
                "type": "transcript",
                "text": transcript.text,
                "duration": round(transcript.duration, 2),
                "cost_usd": round(active.usage.cost_usd, 6),
                "segments": [
                    {
                        "text": seg.text,
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "speaker": seg.speaker or "",
                    }
                    for seg in transcript.final_segments
                ],
            }
        )
        # Tell the client playback is over; it has no other way to know, since
        # a server-driven session has no local end-of-stream.
        enqueue({"type": "demo_finished"})

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
                elif action == "demo":
                    # Server-driven playback so the realtime path is visible
                    # without a microphone. Synthetic audio through the real
                    # session, not an event replay.
                    from koe.evaluation.corpus import ALL_SCRIPTS

                    raw_script = ALL_SCRIPTS.get(str(control.get("meeting", "quarterly-ja")))
                    if raw_script is None:
                        enqueue({"type": "error", "message": "unknown demo meeting"})
                        continue
                    if session is not None:
                        enqueue({"type": "error", "message": "a session is already running"})
                        continue

                    # Both the audio generator and the scripted ASR read the
                    # same re-timed script, so their timelines agree.
                    script = retime(raw_script)
                    language = Language(control.get("language", "ja"))
                    provider = MockASR(
                        script=script,
                        degradation=0.0,
                        timeline=True,
                        name="demo-asr",
                        cost_per_audio_minute_usd=0.006,
                    )
                    session = StreamingSession(
                        ctx, provider, config=SessionConfig(language=language)
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
                    demo_task = asyncio.create_task(run_demo(script), name="koe.demo")
                    continue
                elif action == "minutes":
                    # Generated on the socket the transcript arrived on, so the
                    # client does not have to re-upload a transcript it already
                    # streamed to us.
                    source = last_transcript
                    if source is None or not source.final_segments:
                        enqueue({"type": "error", "message": "no transcript to summarize"})
                        continue
                    try:
                        outcome = await MinutesGenerator(services.llm).generate(source)
                    except Exception as exc:
                        logger.exception("minutes generation failed")
                        enqueue({"type": "error", "message": f"minutes failed: {exc}"})
                        continue
                    services.metrics.increment("minutes.generated")
                    enqueue(
                        {
                            "type": "minutes",
                            "minutes": outcome.minutes.model_dump(mode="json"),
                            "rendered": outcome.minutes.render(),
                            "grounded": outcome.report.grounded,
                            "total_claims": outcome.report.total,
                            "dropped": outcome.dropped,
                            "repairs": outcome.repairs,
                            "cost_usd": round(outcome.usage.cost_usd, 6),
                        }
                    )
                    continue
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
                        _record_spend(services, session)
                        last_transcript = transcript
                        session = None
                    # Deliberately not breaking: the client follows up with a
                    # minutes request on the same socket.
                    continue
                elif action == "close":
                    break
                continue

            if (payload := message.get("bytes")) is not None:
                if session is None:
                    enqueue({"type": "error", "message": "send a start frame first"})
                    continue
                if len(payload) > settings.max_frame_bytes:
                    enqueue({"type": "error", "message": "audio frame too large"})
                    continue
                if session.duration > settings.max_session_seconds:
                    enqueue({"type": "error", "message": "session length limit reached"})
                    break

                # Reject audio arriving faster than realtime by a wide margin.
                # A client streaming a file at speed is not a live caller, and
                # letting it run monopolises a worker and the cost budget.
                elapsed = max(time.monotonic() - session_started, 1e-6)
                if session.duration > elapsed * settings.max_realtime_factor + 5.0:
                    enqueue({"type": "error", "message": "audio is arriving faster than realtime"})
                    services.metrics.increment("ws.rate_limited")
                    break

                try:
                    await session.push_audio(payload)
                except BudgetExceeded as exc:
                    logger.warning("session budget exceeded: %s", exc)
                    enqueue({"type": "error", "message": "session cost limit reached"})
                    break

    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("streaming session failed")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": "internal error"})
    finally:
        if demo_task is not None and not demo_task.done():
            demo_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await demo_task
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump_task
        if session is not None:
            with contextlib.suppress(Exception):
                await session.finish()
            # Audio processed before a disconnect was still paid for. A ledger
            # that only records clean shutdowns under-reports exactly the
            # sessions that went wrong.
            _record_spend(services, session)
        with contextlib.suppress(Exception):
            scope.dispose()
        with contextlib.suppress(Exception):
            await websocket.close()


def _record_spend(services: Services, session: StreamingSession) -> None:
    """Book a finished session's provider spend."""
    if session.usage.cost_usd <= 0 and session.usage.audio_seconds <= 0:
        return
    with contextlib.suppress(BudgetExceeded):
        # The limit was already enforced during the session; recording here
        # must not raise on the teardown path.
        services.ledger.record(session.usage, session_id=session.session_id, modality="asr")
    services.metrics.observe("session.duration_s", session.duration)


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def _web_root() -> Path:
    """Where the built web client lives.

    In a checkout that is ``<repo>/web``, four levels up from this module. In a
    PyInstaller bundle there is no repository: modules report a ``__file__``
    inside the extraction directory, so ``parents[3]`` walks one level *past*
    it and lands on the application folder, where nothing was ever installed.
    The bundle puts the client at ``<_MEIPASS>/web``.

    This shipped broken, and the reason is worth recording: the failure is
    invisible from every endpoint the packaging check exercised. ``/health``,
    ``/v1/*`` and the websocket all worked perfectly in the frozen build — the
    only symptom was that the desktop window contained the "build the client"
    placeholder instead of the application. Verifying a server by asking it
    whether it is alive does not verify that it serves the product.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        return Path(bundle) / "web"
    return Path(__file__).resolve().parents[3] / "web"


def create_app(services: Services | None = None) -> FastAPI:
    """Build the ASGI application."""
    resolved = services or Services.default()
    settings = resolved.settings

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, structured=settings.structured_logs)

        # Refuse to start a misconfigured production process. Failing at boot
        # is far cheaper than discovering a wildcard CORS policy or an
        # unbounded budget from its consequences.
        problems = settings.validate_for_environment()
        if problems:
            for problem in problems:
                logger.error("invalid configuration: %s", problem)
            raise RuntimeError(
                f"refusing to start in {settings.environment}: " + "; ".join(problems)
            )

        logger.info(
            "koe starting",
            extra={
                "version": __version__,
                "environment": settings.environment,
                "mocks": settings.use_mocks,
                "max_sessions": settings.max_concurrent_sessions,
                "japanese_tokenizer": "mecab" if mecab_available() else "character",
            },
        )
        try:
            yield
        finally:
            # Drain in-flight sessions before the process exits, so a deploy
            # does not cut a caller off mid-utterance.
            drained = True
            try:
                async with asyncio.timeout(10.0):
                    await resolved.wait_until_idle()
            except TimeoutError:
                drained = False
            if not drained:
                logger.warning(
                    "shutting down with %d session(s) still active",
                    resolved.active_sessions,
                )
            total = resolved.ledger.total
            logger.info(
                "koe stopped",
                extra={
                    "total_cost_usd": round(total.cost_usd, 6),
                    "audio_seconds": round(total.audio_seconds, 1),
                    "calls": total.calls,
                },
            )

    app = FastAPI(
        title="koe",
        version=__version__,
        description="声 — bilingual JA/EN voice AI harness",
        lifespan=lifespan,
    )
    app.state.services = resolved
    app.add_middleware(RequestContextMiddleware, metrics=resolved.metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
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
