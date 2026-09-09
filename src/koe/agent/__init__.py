"""The chat agent: turns, steps, and the tools a turn calls.

`loop` owns the turn/step vocabulary and the bound on it; `adapters` absorbs
each vendor's opinions about tool calling so the loop never learns which one
it is talking to.
"""

from koe.agent.adapters import (
    AnthropicAdapter,
    MockAdapter,
    OpenAIAdapter,
    adapter_for,
)
from koe.agent.loop import (
    DEFAULT_SYSTEM,
    MAX_STEPS,
    Adapter,
    ChatMessage,
    Conversation,
    ModelReply,
    ToolCall,
    TurnResult,
)

__all__ = [
    "DEFAULT_SYSTEM",
    "MAX_STEPS",
    "Adapter",
    "AnthropicAdapter",
    "ChatMessage",
    "Conversation",
    "MockAdapter",
    "ModelReply",
    "OpenAIAdapter",
    "ToolCall",
    "TurnResult",
    "adapter_for",
]
