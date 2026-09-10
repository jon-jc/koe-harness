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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from koe import __version__
from koe.agent import Conversation, adapter_for
from koe.agent.loop import DEFAULT_SYSTEM
from koe.api.demo import drive_demo, retime
from koe.api.middleware import RequestContextMiddleware
from koe.config import Settings, get_settings
from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Segment, Transcript, attribute_speakers
from koe.harness import AgentRegistry, HarnessAgent
from koe.harness.profile import apply_profile
from koe.harness.profile import load as load_profile
from koe.harness.prompt import SystemPrompt
from koe.kernel.context import Context
from koe.minutes.demo import demo_llm
from koe.minutes.generator import MinutesGenerator
from koe.minutes.schema import Minutes
from koe.pipeline.session import SessionConfig, StreamingSession
from koe.pipeline.vad import VADConfig
from koe.pipeline.vocabulary import user_vocabulary
from koe.plugins import PluginManager
from koe.providers.credentials import (
    PROVIDERS_BY_ID,
    CredentialError,
    CredentialStore,
    Source,
)
from koe.providers.local.resolve import Choice
from koe.providers.mock import MEETING_JA, MockASR, MockDiarization
from koe.providers.verify import verify as verify_credential
from koe.routing.budget import Budget, Priority
from koe.routing.router import Router
from koe.telemetry.ledger import BudgetExceeded, CostLedger
from koe.telemetry.metrics import METRICS, Metrics, configure_logging
from koe.terminal import TerminalFailure, pty_available, terminal_plugin
from koe.text.script import Language
from koe.text.tokenize import mecab_available
from koe.tools import ToolRegistry
from koe.tools.builtin import meeting_tools, workspace_tools, workspace_write_tools
from koe.workspace import Workspace, WorkspaceError, read_before_edit

logger = logging.getLogger(__name__)

#: Live conversations, keyed by id. In-process and deliberately not durable:
#: a chat is a working surface, and persisting one raises questions about
#: where transcripts of a private meeting are stored that a first cut should
#: not answer by accident.
_CONVERSATIONS: dict[str, Any] = {}

#: How many to keep before the oldest is dropped.
MAX_CONVERSATIONS = 32


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
    #: The tool registry every agent surface dispatches through.
    tools: ToolRegistry = field(init=False, repr=False)
    #: File access, fenced to one directory tree.
    workspace: Workspace = field(default_factory=Workspace)
    #: Discovery and lifecycle for everything mounted on the kernel.
    plugins: PluginManager = field(init=False, repr=False)
    #: Live harness agents. One per chat socket, disposed with it.
    agents: AgentRegistry = field(default_factory=AgentRegistry, init=False, repr=False)
    #: The prompt, assembled from whatever is mounted. Plugins contribute
    #: the instructions for the tools they add, so the prompt describes the
    #: harness that exists.
    prompt: SystemPrompt = field(default_factory=SystemPrompt, init=False, repr=False)
    #: The local backends most recently resolved, each with the sentence
    #: explaining the outcome. Cached rather than re-probed, so a credential
    #: change re-runs the policy without sweeping five ports again.
    local_llm: Choice = field(default_factory=Choice, init=False)
    local_asr: Choice = field(default_factory=Choice, init=False)
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
        self._mount_plugins()

    def _mount_plugins(self) -> None:
        """Publish the harness services, then mount everything that uses them.

        Order matters and is the whole reason this is one method: the tool
        registry and the workspace go on the context *first*, because a plugin
        declaring `inject=["tools"]` is only allowed to run once that service
        exists. Mounting the plugins first would leave every one of them
        waiting for a service that arrives a line later.
        """
        self.tools = ToolRegistry(self.ctx)
        self.ctx.provide("tools", self.tools, replace=True)
        self.ctx.provide("prompt", self.prompt, replace=True)
        self._mount_prompt()
        self.ctx.provide("workspace", self.workspace, replace=True)

        from koe.desktop.paths import config_dir, data_dir

        self.plugins = PluginManager(
            self.ctx,
            directory=data_dir() / "plugins",
            state=config_dir() / "plugins.json",
        )
        # First-party tool groups are plugins like any other, and can be
        # turned off from the same panel.
        self.plugins.add_builtin(
            "workspace-tools",
            workspace_tools,
            description="Read-only file access for the assistant: list, read, glob, grep.",
            inject=("tools", "workspace"),
        )
        # Order matters for readability rather than correctness — the kernel
        # activates on dependencies, not on registration order — but mounting
        # the guard immediately before the tools it guards is how the listing
        # reads to whoever opens the plugins panel.
        self.plugins.add_builtin(
            "read-before-edit",
            read_before_edit,
            description=(
                "Refuses a write to a file the assistant has not read, and a "
                "write to one that changed since. Turning this off leaves the "
                "write tools unguarded."
            ),
            inject=("tools", "workspace"),
        )
        self.plugins.add_builtin(
            "workspace-write-tools",
            workspace_write_tools,
            description="Lets the assistant write and edit files in the workspace.",
            inject=("tools", "workspace"),
        )
        self.plugins.add_builtin(
            "meeting-tools",
            meeting_tools,
            description="Lets the assistant read this session's transcript and 議事録.",
            inject=("tools",),
        )
        self.plugins.add_builtin(
            "user-vocabulary",
            user_vocabulary,
            description=(
                "Corrects names, acronyms and product words the recognizer has "
                "no way to know, from an editable word list. Turning this off "
                "leaves transcripts exactly as the provider returned them."
            ),
        )
        self.plugins.add_builtin(
            "terminal",
            terminal_plugin,
            description=(
                "Persistent shell sessions, for the terminal panel and for the "
                "assistant. Turning this off removes both."
            ),
            inject=("tools",),
        )
        self.plugins.discover()
        self._apply_profile()
        self.plugins.activate_all()

    def _apply_profile(self) -> None:
        """Let a profile narrow or retune the built-in composition.

        Applied after discovery and before activation, which is the only window
        where a row can still change what starts. A missing profile is an empty
        one: koe runs its built-in composition, and a profile only ever adjusts
        it, so "no file" and "a file that changes nothing" behave the same.
        """
        from koe.desktop.paths import config_dir

        try:
            profile = load_profile(
                config_dir() / "profile.toml",
                patches=[config_dir() / "profile.patch.toml"],
            )
            if not len(profile):
                return
            for change in apply_profile(self.plugins, profile):
                logger.info("profile: %s", change)
        except Exception:
            # A broken profile must not stop the process: koe's built-in
            # composition is a working one, and starting with it plus a logged
            # complaint beats refusing to start at all.
            logger.exception("profile could not be applied; using the built-in composition")

    def _mount_prompt(self) -> None:
        """The sections koe always has, and the variables they reference.

        Tool-specific instructions belong to the plugins that register the
        tools -- `workspace_tools` contributes the section describing them --
        so turning a plugin off removes both the tool and the paragraph telling
        the model to use it.
        """
        self.prompt.section(
            "koe:identity",
            "You are the assistant inside koe, a bilingual (Japanese and English) "
            "voice AI harness that records meetings, transcribes them with speaker "
            "labels, and generates verified 議事録.",
            order="IDENTITY",
        )
        self.prompt.section(
            "koe:language",
            "Answer in the language the user writes in. Be concise and concrete. "
            "When you have used a tool, say what you found rather than narrating "
            "that you used it.",
            order="LANGUAGE",
        )
        self.prompt.section(
            "koe:tools",
            # Dynamic: it names the tools that are actually mounted right now,
            # so a disabled plugin stops being advertised without anyone having
            # to remember to edit this.
            lambda: (
                (
                    "Tools available to you: "
                    + ", ".join(sorted(spec.name for spec in self.tools))
                    + ". Prefer using a tool over guessing: if a question is about the "
                    "code or the meeting, look."
                )
                if len(self.tools)
                else "You have no tools in this session; answer from the conversation alone."
            ),
            order="TOOL_WORKSPACE",
        )

    def refresh_llm(self) -> str:
        """Point the LLM service at whichever backend should be serving.

        Called at startup and again whenever a key is added or removed, so a
        user who pastes a key gets the real model on their next request rather
        than after a restart. Registering through the context rather than only
        assigning the attribute is what lets dependent plugins rebuild against
        the new client — the reason the kernel tracks services reactively.

        Synchronous, and deliberately: the local backend is *resolved*
        asynchronously by :meth:`rescan_local` because it probes the network,
        but the policy that picks between local, cloud and mock is a pure
        function of what has already been found. Keeping the two apart is what
        lets a credential change re-run the policy without re-probing five
        ports.
        """
        chosen = "mock"
        provider: Any = demo_llm()

        # Asked for explicitly. A local model that is actually running wins over
        # a key, which is the whole point of the setting: a privacy-motivated
        # user should not have to delete their credentials to stop them being
        # used.
        if (
            not self.settings.force_mock_providers
            and self.settings.prefer_local_llm
            and self.local_llm
        ):
            self.llm = self.local_llm.provider
            self.ctx.provide("llm", self.llm, replace=True)
            logger.info("llm provider selected", extra={"provider": "local"})
            return "local"

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

        if chosen == "mock" and not self.settings.force_mock_providers and self.local_llm:
            # No key, but something is running on this machine. Better than the
            # mock by a wide margin, and it costs nothing to use.
            provider = self.local_llm.provider
            chosen = "local"

        self.llm = provider
        self.ctx.provide("llm", provider, replace=True)
        logger.info("llm provider selected", extra={"provider": chosen})
        return chosen

    async def rescan_local(self) -> str:
        """Probe for local backends, then re-run the selection policy.

        The only async part of provider selection, because it is the only part
        that asks the network a question. Called at startup and whenever the
        user changes a local setting.
        """
        from koe.providers.local.resolve import resolve_asr, resolve_llm

        if self.settings.force_mock_providers:
            # The demo path stays deterministic. Probing here would make CI's
            # behaviour depend on what happens to be listening on the runner.
            self.local_llm = Choice(reason="mock providers are forced")
            self.local_asr = Choice(reason="mock providers are forced")
            return self.refresh_llm()

        self.local_llm = await resolve_llm(self.settings)
        self.local_asr = resolve_asr(self.settings)
        self.apply_local_asr()
        logger.info(
            "local backends resolved",
            extra={"llm": self.local_llm.reason, "asr": self.local_asr.reason},
        )
        return self.refresh_llm()

    def apply_local_asr(self) -> None:
        """Register or remove local Whisper on the ASR router.

        Registered *alongside* the other backends rather than replacing them,
        because that is what the router is for: local Whisper is free and slow,
        a hosted vendor is quick and metered, and which one a given request
        should use is exactly the trade the router exists to make. Switching
        local recognition on narrows the choice rather than removing it.
        """
        name = "local-whisper"
        if self.local_asr:
            self.asr_router.register(self.local_asr.provider)
        else:
            self.asr_router.unregister(name)

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


class LocalUpdate(BaseModel):
    """Local-model settings. Every field optional: the panel sends deltas.

    None means "leave this alone", which is what lets one control change one
    thing without the others racing it — a panel that sent the whole object
    would have the ASR toggle undo an in-flight base URL edit.
    """

    base_url: str | None = Field(default=None, max_length=2_048)
    model: str | None = Field(default=None, max_length=256)
    api_key: str | None = Field(default=None, max_length=512)
    prefer: bool | None = None
    asr_enabled: bool | None = None
    asr_model: str | None = Field(default=None, max_length=64)
    asr_device: str | None = Field(default=None, pattern="^(auto|cpu|cuda)$")


class VocabularyUpdate(BaseModel):
    """The user's word list, in the import format.

    One field rather than a structured list of entries: the format is designed
    to be typed and pasted, and round-tripping it through JSON objects would
    make the editable thing and the stored thing two different things.
    """

    text: str = Field(default="", max_length=100_000)


class ChatRequest(BaseModel):
    """One turn of conversation."""

    message: str = Field(min_length=1, max_length=32_000)
    #: Continues an existing conversation; omit to start one.
    conversation: str = ""


class TerminalOpen(BaseModel):
    """Open a terminal session."""

    name: str = ""


class TerminalSend(BaseModel):
    """Write to a session and wait for it to settle."""

    text: str
    #: Bounded so a client cannot pin a request thread indefinitely.
    timeout_s: float = Field(default=30.0, ge=0.5, le=300.0)


class PluginToggle(BaseModel):
    """Turn one plugin on or off."""

    enabled: bool


class ToolCallRequest(BaseModel):
    """Arguments for a direct tool invocation."""

    arguments: dict[str, Any] = Field(default_factory=dict)


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

    # ----------------------------------------------------------------- plugins

    @router.get("/v1/plugins")
    async def list_plugins() -> JSONResponse:
        """Everything mounted on the kernel, first-party tools included."""
        return JSONResponse(services.plugins.to_dict())

    # ------------------------------------------------------------ local models

    def _local_payload() -> dict[str, Any]:
        from koe.providers.local.endpoints import WELL_KNOWN
        from koe.providers.local.whisper import SIZES
        from koe.providers.local.whisper import available as whisper_available

        settings_now = services.settings
        return {
            "llm": {
                "active": bool(services.local_llm),
                "reason": services.local_llm.reason,
                "base_url": settings_now.local_llm_base_url,
                "model": settings_now.local_llm_model,
                "prefer": settings_now.prefer_local_llm,
                # Whether the *serving* provider is the local one, which is not
                # the same as one being available: a configured API key wins
                # unless `prefer` says otherwise, and the panel must not claim
                # a model is in use when it is merely ready.
                "in_use": getattr(services.llm, "base_url", "") != "",
            },
            "asr": {
                "active": bool(services.local_asr),
                "reason": services.local_asr.reason,
                "enabled": settings_now.local_asr_enabled,
                "installed": whisper_available(),
                "model": settings_now.local_asr_model,
                "device": settings_now.local_asr_device,
                "sizes": [
                    {
                        "id": size.id,
                        "label": size.label,
                        "download_mb": size.download_mb,
                        "typical_rtf": size.typical_rtf,
                        "suitable_ja": Language.JA in size.suitable,
                        "suitable_en": Language.EN in size.suitable,
                    }
                    for size in SIZES
                ],
            },
            "known_servers": [
                {"id": s.id, "label": s.label, "port": s.port, "docs_url": s.docs_url}
                for s in WELL_KNOWN
            ],
        }

    @router.get("/v1/local")
    async def local_status() -> JSONResponse:
        """What local inference is available, and what is actually serving."""
        return JSONResponse(_local_payload())

    @router.post("/v1/local/discover")
    async def local_discover() -> JSONResponse:
        """Sweep the well-known ports and report what answered.

        A POST because it is not free — it opens sockets — and because the
        panel triggers it from a button rather than on every render.
        """
        from koe.providers.local.discovery import discover

        sweep = await discover()
        return JSONResponse(sweep.to_dict())

    @router.put("/v1/local")
    async def local_update(request: LocalUpdate) -> JSONResponse:
        """Change local-model settings and re-resolve.

        Settings are mutated on the live object rather than persisted: koe
        reads configuration from the environment, and a panel that silently
        wrote a config file would make the environment stop being the answer to
        "why is it doing that". The panel says the choice lasts for this run.
        """
        settings_now = services.settings
        if request.base_url is not None:
            settings_now.local_llm_base_url = request.base_url.strip()
        if request.model is not None:
            settings_now.local_llm_model = request.model.strip()
        if request.api_key is not None:
            settings_now.local_llm_api_key = request.api_key.strip()
        if request.prefer is not None:
            settings_now.prefer_local_llm = request.prefer
        if request.asr_enabled is not None:
            settings_now.local_asr_enabled = request.asr_enabled
        if request.asr_model is not None:
            from koe.providers.local.whisper import SIZES_BY_ID

            if request.asr_model not in SIZES_BY_ID:
                raise HTTPException(status_code=400, detail=f"unknown size {request.asr_model!r}")
            settings_now.local_asr_model = request.asr_model
        if request.asr_device is not None:
            settings_now.local_asr_device = request.asr_device

        await services.rescan_local()
        return JSONResponse(_local_payload())

    # -------------------------------------------------------------- vocabulary

    def _vocabulary_payload(store: Any) -> dict[str, Any]:
        if store is None:
            # The plugin is off. Reported rather than 404'd: the client wants to
            # show the panel with an explanation, and a missing endpoint and a
            # disabled feature are different things.
            return {"enabled": False, "text": "", "entries": 0, "terms": []}
        return {
            "enabled": True,
            "text": store.text(),
            "entries": len(store),
            "terms": list(store.terms),
        }

    @router.get("/v1/vocabulary")
    async def get_vocabulary() -> JSONResponse:
        """The user's word list."""
        return JSONResponse(_vocabulary_payload(services.ctx.get("vocabulary")))

    @router.put("/v1/vocabulary")
    async def put_vocabulary(request: VocabularyUpdate) -> JSONResponse:
        """Replace the word list. Takes effect on the next utterance."""
        store = services.ctx.get("vocabulary")
        if store is None:
            raise HTTPException(status_code=409, detail="the user-vocabulary plugin is disabled")
        try:
            store.save(request.text)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not save: {exc}") from exc
        return JSONResponse(_vocabulary_payload(store))

    @router.put("/v1/plugins/{name}")
    async def set_plugin_enabled(name: str, request: PluginToggle) -> JSONResponse:
        """Enable or disable a plugin, taking effect immediately.

        Disabling unmounts: the plugin's tools, listeners and services are gone
        when this returns, not merely marked inactive.
        """
        try:
            record = services.plugins.set_enabled(name, request.enabled)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown plugin {name!r}") from None
        services.metrics.increment(f"plugins.{'enabled' if request.enabled else 'disabled'}")
        return JSONResponse({"plugin": record.to_dict(), "tools": services.tools.describe()})

    @router.post("/v1/plugins/reload")
    async def reload_plugins() -> JSONResponse:
        """Re-scan the plugins directory, unmounting anything that disappeared."""
        services.plugins.reload()
        return JSONResponse(services.plugins.to_dict())

    # ------------------------------------------------------------------- tools

    @router.get("/v1/tools")
    async def list_tools() -> JSONResponse:
        """Registered tools, with the host-only fields an operator needs."""
        return JSONResponse({"tools": services.tools.describe()})

    @router.post("/v1/tools/{name}")
    async def call_tool(name: str, request: ToolCallRequest) -> JSONResponse:
        """Run one tool directly.

        Exists so a tool can be exercised without a model in the loop, which is
        what makes a misbehaving tool debuggable. It runs the same guarded
        pipeline, so a policy that would refuse the model refuses this too.
        """
        result = await services.tools.call(name, request.arguments, owner="operator")
        services.metrics.increment(f"tools.{'ok' if result.ok else 'error'}")
        return JSONResponse(result.to_dict())

    # -------------------------------------------------------------------- chat

    def _conversation(conversation_id: str) -> Any:
        """Fetch or start a conversation, bound to the LLM configured *now*.

        The adapter is chosen per turn rather than held for the life of the
        conversation, so pasting an API key mid-chat takes effect on the next
        message instead of after a restart — the same property the rest of the
        app holds to.
        """
        existing = _CONVERSATIONS.get(conversation_id) if conversation_id else None
        if existing is not None:
            existing.adapter = adapter_for(services.llm)
            return existing

        conversation = Conversation(adapter_for(services.llm), services.tools)
        _CONVERSATIONS[conversation.id] = conversation
        # Bounded: a long-running desktop app must not accumulate every
        # conversation anyone ever started. Oldest first, which is insertion
        # order for a dict.
        while len(_CONVERSATIONS) > MAX_CONVERSATIONS:
            _CONVERSATIONS.pop(next(iter(_CONVERSATIONS)))
        return conversation

    @router.get("/v1/chat/{conversation_id}")
    async def chat_history(conversation_id: str) -> JSONResponse:
        conversation = _CONVERSATIONS.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="no such conversation")
        return JSONResponse({"conversation": conversation.id, "messages": conversation.history()})

    @router.post("/v1/chat")
    async def chat(request: ChatRequest) -> JSONResponse:
        """One turn, answered when it is finished.

        The websocket below is the surface the UI uses; this exists for
        scripting and for anything that would rather have one response than a
        stream of events.
        """
        conversation = _conversation(request.conversation)
        result = await conversation.send(request.message)
        services.metrics.increment("chat.turns")
        return JSONResponse({"conversation": conversation.id, **result.to_dict()})

    # ---------------------------------------------------------------- terminal

    def _terminals() -> Any:
        """The terminal service, or a 503 saying why there isn't one.

        503 rather than 404: the endpoint exists and the capability is simply
        not mounted, which is a different thing for a client to show than a
        path that was never real.
        """
        service = services.ctx.get("terminals")
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="the terminal plugin is not enabled",
            )
        return service

    @router.get("/v1/terminal/capabilities")
    async def terminal_capabilities() -> JSONResponse:
        """What kinds of terminal this machine can actually open.

        The client asks before choosing a panel: a pty gives full-screen
        programs and needs an emulator, pipes give clean text and do not.
        Guessing from the platform would be wrong on a Windows box without
        pywinpty installed.
        """
        return JSONResponse(
            {
                "pty": pty_available() and services.ctx.get("terminals") is not None,
                "shell": services.ctx.get("terminals") is not None,
            }
        )

    @router.get("/v1/terminal/sessions")
    async def list_terminals() -> JSONResponse:
        service = _terminals()
        return JSONResponse(
            {"sessions": [session.to_dict() for session in service.list(owner="ui")]}
        )

    @router.post("/v1/terminal/sessions")
    async def open_terminal(request: TerminalOpen) -> JSONResponse:
        service = _terminals()
        try:
            session = await service.open(owner="ui", name=request.name)
        except TerminalFailure as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(session.to_dict())

    @router.post("/v1/terminal/sessions/{session_id}/send")
    async def send_to_terminal(session_id: str, request: TerminalSend) -> JSONResponse:
        service = _terminals()
        try:
            outcome = await service.send(
                session_id, request.text, owner="ui", timeout_s=request.timeout_s
            )
        except TerminalFailure as exc:
            # 409 for "your request conflicts with the session's state" —
            # a send already running, a shell that exited — and 404 for a
            # session that is not there. Both are the client's to act on.
            status = 404 if exc.code.value in {"no_session", "foreign_session"} else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return JSONResponse(outcome.to_dict())

    @router.get("/v1/terminal/sessions/{session_id}/output")
    async def read_terminal(session_id: str) -> JSONResponse:
        """Output since the last read. Polled by the panel while a command runs."""
        service = _terminals()
        try:
            return JSONResponse({"output": service.read(session_id, owner="ui")})
        except TerminalFailure as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.delete("/v1/terminal/sessions/{session_id}")
    async def close_terminal(session_id: str) -> JSONResponse:
        service = _terminals()
        try:
            session = await service.close(session_id, owner="ui")
        except TerminalFailure as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(session.to_dict())

    # --------------------------------------------------------------- workspace

    @router.get("/v1/fs/tree")
    async def fs_tree(path: str = "") -> JSONResponse:
        """One directory of the workspace."""
        try:
            entries = services.workspace.list_dir(path)
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {
                "path": path,
                "entries": [entry.to_dict() for entry in entries],
                **services.workspace.info(),
            }
        )

    @router.get("/v1/fs/file")
    async def fs_file(path: str) -> JSONResponse:
        """One file, bounded, with the language token a highlighter wants."""
        try:
            view = services.workspace.read(path)
        except WorkspaceError as exc:
            # 404 for a missing file, 400 for one we refuse to serve: the
            # client shows a different thing for "gone" than for "binary".
            status = 404 if exc.code == "not_found" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return JSONResponse(view.to_dict())

    @router.get("/v1/fs/search")
    async def fs_search(q: str, glob: str = "") -> JSONResponse:
        """Regex search across the workspace."""
        try:
            hits = services.workspace.grep(q, glob=glob or None)
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"query": q, "hits": hits})

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


async def _pty_endpoint(websocket: WebSocket, services: Services) -> None:
    """Bridge one websocket to one pty session for as long as both live."""
    await websocket.accept()

    service = services.ctx.get("terminals")
    if service is None:
        await websocket.send_json({"t": "fatal", "message": "the terminal plugin is not enabled"})
        await websocket.close()
        return
    if not pty_available():
        # Named rather than generic: the remedy is a pip install, and a user
        # who is told "unavailable" has no way to discover that.
        await websocket.send_json(
            {
                "t": "fatal",
                "code": "no_pty",
                "message": (
                    "no pty on this machine — install pywinpty for full-screen "
                    "programs, or use the shell terminal"
                ),
            }
        )
        await websocket.close()
        return

    session = None
    pump: asyncio.Task[None] | None = None
    try:
        opening = await websocket.receive_json()
        cols = max(20, min(500, int(opening.get("cols") or 80)))
        rows = max(5, min(200, int(opening.get("rows") or 24)))

        session = await service.open(owner="pty-ui", type="pty", name="panel", size=(cols, rows))
        await websocket.send_json(
            {"t": "ready", "id": session.id, "pid": session.channel.pid, "cwd": session.cwd}
        )

        async def drain() -> None:
            """Forward output as it arrives.

            Waits on the session rather than reading the channel: the service
            already runs exactly one reader, and a second one steals chunks
            from the first. That bug does not look like a bug — output still
            arrives, just not all of it — which is why this goes through the
            session even though reading the channel would be one line shorter.
            """
            while session.alive:
                chunk = await session.follow(wait_s=30.0)
                if chunk:
                    await websocket.send_json({"t": "o", "d": chunk})
            # Whatever landed between the last wait and the exit.
            tail = session.drain()
            if tail:
                await websocket.send_json({"t": "o", "d": tail})
            await websocket.send_json({"t": "exit", "code": session.channel.exit_code})

        pump = asyncio.create_task(drain())

        while True:
            frame = await websocket.receive_json()
            kind = frame.get("t")
            if kind == "i":
                await session.channel.write(str(frame.get("d", "")).encode("utf-8"))
            elif kind == "r":
                session.channel.resize(
                    max(20, min(500, int(frame.get("cols") or cols))),
                    max(5, min(200, int(frame.get("rows") or rows))),
                )
    except WebSocketDisconnect:
        # The normal end: the pane closed or the tab went away.
        return
    except Exception:
        logger.exception("pty socket failed")
    finally:
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
        if session is not None:
            # The pty dies with its socket. A session nobody can reach is a
            # leaked shell, and there is no id here for anyone to reconnect
            # with — the pipe backend is the one that survives a reload.
            with contextlib.suppress(Exception):
                await service.close(session.id, owner="pty-ui")
        with contextlib.suppress(Exception):
            await websocket.close()


async def _chat_endpoint(websocket: WebSocket, services: Services) -> None:
    """Drive one harness agent over a socket until the client goes away.

    **The socket never blocks on a turn**, and that is the whole difference
    between this and the chat it replaces. The old endpoint awaited
    `conversation.send(text)`, so between asking and being answered it could
    not hear anything — which meant the two things a person most wants while a
    model is working, *stop* and *no, look over there*, were unreachable by
    construction.

    Here the receive loop only ever receives. Input is queued on the agent's
    inbox and a driver task consumes it, so a frame arriving mid-turn is
    ordinary rather than a race.

    Four frames:

    ``{"message": "..."}``  a prompt, taking its own turn
    ``{"steer": "..."}``    join the running turn at its next step boundary
    ``{"cancel": true}``    stop the current turn, keeping anything queued
    ``{"sync": <seq>}``     replay session events after `seq`, for a reconnect
    """
    await websocket.accept()
    handle: Any = None

    async def emit(event: str, payload: dict[str, Any]) -> None:
        await websocket.send_json({"type": event, **payload})

    def build() -> Any:
        agent = HarnessAgent(
            adapter_for(services.llm),
            services.tools,
            system=DEFAULT_SYSTEM,
            prompt=services.prompt,
            on_event=emit,
        )
        return services.agents.create(agent)

    try:
        while True:
            frame = await websocket.receive_json()

            if frame.get("cancel"):
                if handle is not None:
                    # The inbox survives: someone who stops one answer usually
                    # still wants the things they queued behind it.
                    handle.agent.cancel("user", keep_inbox=True)
                    await emit("cancelled", {"agent": handle.agent.id})
                continue

            if handle is not None and frame.get("commands"):
                await emit("commands", {"commands": handle.agent.commands.describe()})
                continue

            if handle is not None and frame.get("sync") is not None:
                since = int(frame.get("sync") or 0)
                await emit(
                    "session",
                    {
                        "session": handle.agent.session.id,
                        "events": [e.to_dict() for e in handle.agent.session.since(since)],
                        "last_seq": handle.agent.session.last_seq,
                    },
                )
                continue

            steer = str(frame.get("steer", "")).strip()
            text = str(frame.get("message", "")).strip()
            if not steer and not text:
                await websocket.send_json({"type": "error", "message": "empty message"})
                continue

            if handle is None or frame.get("reset"):
                if handle is not None:
                    handle.dispose()
                handle = build()
                await emit(
                    "conversation",
                    {
                        "id": handle.agent.id,
                        "session": handle.agent.session.id,
                        "adapter": handle.agent.adapter.name,
                    },
                )
            else:
                # Re-resolved per message, so a key pasted mid-conversation
                # takes effect on the next turn rather than after a restart.
                handle.agent.adapter = adapter_for(services.llm)

            # A slash command is an instruction to the harness, not something
            # the user said to the model, so it runs instead of a turn and
            # never reaches the inbox.
            outcome = await handle.agent.command(steer or text)
            if outcome is not None:
                await emit("command", outcome.to_dict())
                continue

            if steer:
                handle.agent.steer(steer)
                await emit("steered", {"text": steer, "turn": handle.agent.status["turn"]})
            else:
                handle.agent.followup(text)
            services.metrics.increment("chat.turns")
    except WebSocketDisconnect:
        # The normal way a chat ends: the tab closed.
        return
    except Exception:
        logger.exception("chat socket failed")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": "internal error"})
    finally:
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.dispose()
        with contextlib.suppress(Exception):
            await websocket.close()


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
        _publish_meeting(services, transcript)
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
                            partial_interval_ms=_clamp(
                                control.get("partial_interval_ms"), 150.0, 2000.0, 500.0
                            ),
                            vad=_vad_overrides(control, language),
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
                    services.ctx.provide(
                        "last_minutes", outcome.minutes.model_dump(mode="json"), replace=True
                    )
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
                        _publish_meeting(services, transcript)
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


def _publish_meeting(services: Services, transcript: Any) -> None:
    """Make the finished meeting readable by tools and plugins.

    Published on the context rather than held on `Services` so a plugin can
    reach it through the same `ctx.get` every other service uses — and so a
    plugin that wants to react can watch the service appear instead of
    polling. `replace=True` because a second meeting supersedes the first.
    """
    lines = [
        f"[{seg.start:.0f}s] {seg.speaker or '-'}: {seg.text}" for seg in transcript.final_segments
    ]
    services.ctx.provide("last_transcript", "\n".join(lines), replace=True)
    services.ctx.provide(
        "last_segments",
        [
            {
                "text": seg.text,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": seg.speaker or "",
            }
            for seg in transcript.final_segments
        ],
        replace=True,
    )


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    """Coerce a client-supplied number into a range we are willing to run.

    Session settings are user-facing controls now, which means they arrive
    from a browser and cannot be trusted to be sane: a `silence_to_end_ms` of
    zero ends every utterance on the first quiet frame, and one of 60000 means
    a session that never emits. Clamping beats validating-and-rejecting here
    because the sliders that produce these values are already bounded, so an
    out-of-range value is a bug or an attack rather than a user's intent.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


def _vad_overrides(control: dict[str, Any], language: Language) -> VADConfig | None:
    """Endpointing settings from the client, over the language default.

    Returns None when the client sent nothing, so the language-tuned default
    stays in force — Japanese needs a longer silence window than English, and
    silently replacing that with a hardcoded number would undo the single
    most user-visible tuning decision in the realtime path.
    """
    silence = control.get("silence_to_end_ms")
    threshold = control.get("speech_threshold_db")
    if silence is None and threshold is None:
        return None

    base = VADConfig.for_language(language)
    return replace(
        base,
        silence_to_end_ms=_clamp(silence, 200.0, 3000.0, base.silence_to_end_ms),
        speech_threshold_db=_clamp(threshold, 3.0, 24.0, base.speech_threshold_db),
    )


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

        # Probing happens at startup rather than on the first request, so a
        # user who already has Ollama running finds koe using it instead of
        # finding a settings panel and a decision to make. Failure here is a
        # log line: a machine with no local models is the ordinary case, not
        # a broken one.
        try:
            await resolved.rescan_local()
        except Exception:
            logger.exception("local backend discovery failed")

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

    @app.websocket("/v1/terminal/pty")
    async def terminal_pty(websocket: WebSocket) -> None:
        """A real terminal, streamed.

        REST polling is right for the pipe backend, where a command settles and
        returns a block of text. It is wrong for a pty: `vim` redraws on every
        keystroke, so the round trip has to be a frame rather than a request,
        and the connection has to carry a window size that changes when the
        pane does.
        """
        await _pty_endpoint(websocket, resolved)

    @app.websocket("/v1/chat/stream")
    async def chat_stream(websocket: WebSocket) -> None:
        """A turn, streamed as it happens.

        The turn emits before it returns — step started, tool called, tool
        returned — so the panel shows a model reading three files rather than
        a spinner that resolves into a paragraph. A request/response endpoint
        can only be watched by waiting, and a turn that calls tools is exactly
        the case where waiting is longest and the intermediate steps are the
        most interesting thing on screen.
        """
        await _chat_endpoint(websocket, resolved)

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
