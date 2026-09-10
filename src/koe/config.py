"""Runtime configuration.

Everything operational is settable from the environment, because the thing that
distinguishes a service you can run from a service you can *operate* is whether
you can change its behaviour without a rebuild. Concurrency caps, budgets,
timeouts and the endpointing window all move between staging and production,
and none of them should require a code change to move.

Two rules the settings enforce:

**Validate at startup, not at first use.** A malformed ``KOE_MAX_SESSIONS``
should stop the process immediately with a clear message, not surface forty
minutes later as a confusing failure on one request. Pydantic does that here.

**Ship safe defaults.** Every limit has a value that is sane for a small
production deployment, so an operator who sets nothing still gets bounded
concurrency, bounded session length and a cost ceiling. Unlimited-by-default is
how a demo becomes an incident.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from koe.text.script import Language


class Settings(BaseSettings):
    """Process configuration, read from ``KOE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="KOE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- service ---------------------------------------------------------
    environment: Literal["local", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: JSON logs in a container, human-readable locally.
    structured_logs: bool = True

    # -- limits ----------------------------------------------------------
    #: Concurrent streaming sessions. Each holds an audio buffer and a decoder,
    #: so this is the main lever on memory per task.
    max_concurrent_sessions: int = Field(default=64, ge=1, le=10_000)
    #: Hard cap on a single session. A stuck client that never disconnects is
    #: otherwise an unbounded cost and an unbounded buffer.
    max_session_seconds: float = Field(default=4 * 60 * 60, gt=0)
    #: Largest accepted audio frame. A 20ms frame is 640 bytes; anything near
    #: this is not audio.
    max_frame_bytes: int = Field(default=1 << 20, ge=1024)
    #: Per-connection ceiling on inbound audio rate, as a multiple of realtime.
    #: A client sending faster than this is either broken or replaying a file
    #: at speed, and either way it should not be allowed to monopolise a worker.
    max_realtime_factor: float = Field(default=4.0, gt=0)

    # -- cost ------------------------------------------------------------
    session_budget_usd: float | None = Field(default=5.0, gt=0)
    tenant_budget_usd: float | None = Field(default=None, gt=0)

    # -- behaviour -------------------------------------------------------
    default_language: Language = Language.JA
    partial_interval_ms: float = Field(default=500.0, ge=100.0, le=5_000.0)
    #: Silence before an utterance is considered over. Overrides the
    #: language-specific default when set, for operators tuning a deployment
    #: against their own audio.
    silence_to_end_ms: float | None = Field(default=None, ge=100.0, le=5_000.0)

    # -- providers -------------------------------------------------------
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    #: Use the deterministic mocks even when credentials are present. The demo
    #: path, and the way CI runs the server without spending money.
    force_mock_providers: bool = False

    llm_model: str = "claude-opus-5"
    asr_model: str = "whisper-large-v3"

    # -- local models ----------------------------------------------------
    # Nothing here needs a key, which is the point: a machine with Ollama
    # running and no credentials at all is a complete koe install.
    #
    #: Probe the well-known ports at startup. On by default, because finding
    #: the Ollama someone already runs is the entire point -- and off in the
    #: test suite, where what happens to be listening on the machine must not
    #: decide what the tests exercise. An explicit `local_llm_base_url` is
    #: honoured either way: this gates the sweep, not the feature.
    discover_local_models: bool = True
    #: Base URL of an OpenAI-compatible server on this machine. Empty means
    #: "look for one" -- koe probes the well-known ports rather than asking
    #: someone to know which port their GUI chose.
    local_llm_base_url: str = ""
    #: Which model to ask that server for. Empty means the first it offers,
    #: which is the right guess when exactly one is loaded and a harmless one
    #: when several are.
    local_llm_model: str = ""
    #: Some local servers sit behind a proxy that wants a bearer token.
    local_llm_api_key: str = ""
    #: Prefer a discovered local model over a configured cloud key. Off by
    #: default: someone who has pasted an API key has expressed a preference,
    #: and silently overriding it because a server happens to be listening
    #: would be the harness making a policy decision on their behalf.
    prefer_local_llm: bool = False

    #: Run speech recognition on this machine. The setting that decides
    #: whether audio leaves the device at all.
    local_asr_enabled: bool = False
    #: Whisper checkpoint. See koe.providers.local.whisper.SIZES -- the small
    #: ones are not suitable for Japanese and koe says so rather than letting
    #: it be discovered from a meeting transcript.
    local_asr_model: str = "large-v3-turbo"
    #: "auto", "cpu" or "cuda".
    local_asr_device: str = "auto"
    #: CTranslate2 quantization. "default" lets it choose for the device.
    local_asr_compute_type: str = "default"

    # -- http ------------------------------------------------------------
    #: CORS origins. The wildcard default is fine for a local demo and is
    #: rejected in production by the validator below.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard_in_production(cls, value: list[str], info: object) -> list[str]:
        # Checked again in `validate_for_environment`, which has the full model.
        return value

    def validate_for_environment(self) -> list[str]:
        """Return configuration problems that matter for this environment.

        Kept separate from field validation because these are cross-field
        rules, and because a local developer should not be blocked by a policy
        that only makes sense in production.
        """
        problems: list[str] = []
        if self.environment != "production":
            return problems

        if "*" in self.cors_origins:
            problems.append(
                "cors_origins must not be '*' in production; set KOE_CORS_ORIGINS "
                "to the exact origins that serve the client"
            )
        if not self.structured_logs:
            problems.append("structured_logs should be enabled in production")
        if self.session_budget_usd is None:
            problems.append(
                "session_budget_usd must be set in production; an unbounded "
                "session is an unbounded bill"
            )
        if self.force_mock_providers:
            problems.append("force_mock_providers is enabled in production")
        return problems

    @property
    def use_mocks(self) -> bool:
        """Whether to run on deterministic mocks.

        True when explicitly forced, or when no credentials exist -- so a fresh
        checkout starts and serves rather than crashing on a missing key.
        """
        if self.force_mock_providers:
            return True
        return not (self.anthropic_api_key or self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, parsed once."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. For tests that manipulate the environment."""
    get_settings.cache_clear()
