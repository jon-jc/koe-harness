"""Provider adapters behind narrow protocols.

Every backend -- local Whisper, a managed ASR vendor, Claude, GPT -- implements
the same protocol and publishes a :class:`ProviderInfo` describing its cost,
speed and expected quality. That metadata is what lets the router choose
between them at runtime instead of the choice being frozen into a deployment.
"""

from koe.providers.base import (
    ASRProvider,
    DiarizationProvider,
    LLMProvider,
    LLMResponse,
    Message,
    Modality,
    ProviderError,
    ProviderInfo,
    ProviderTimeout,
    StreamingASRProvider,
    Usage,
    measured,
)
from koe.providers.mock import (
    MEETING_EN,
    MEETING_JA,
    MEETING_MIXED,
    MockASR,
    MockDiarization,
    MockLLM,
    ScriptedUtterance,
)

__all__ = [
    "MEETING_EN",
    "MEETING_JA",
    "MEETING_MIXED",
    "ASRProvider",
    "DiarizationProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MockASR",
    "MockDiarization",
    "MockLLM",
    "Modality",
    "ProviderError",
    "ProviderInfo",
    "ProviderTimeout",
    "ScriptedUtterance",
    "StreamingASRProvider",
    "Usage",
    "measured",
]
