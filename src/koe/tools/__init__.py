"""Tool definitions and the guarded pipeline that runs them.

A tool is the unit an LLM can act with. This package owns what one is, how the
registry projects it for a model, and the three-phase pipeline where policy
attaches — see :mod:`koe.tools.registry` for why each of those is shaped the
way it is.
"""

from koe.tools.registry import (
    MAX_RESULT_CHARS,
    Denial,
    ToolError,
    ToolInvocationError,
    ToolRegistry,
    ToolResult,
    ToolRun,
    ToolSpec,
)

__all__ = [
    "MAX_RESULT_CHARS",
    "Denial",
    "ToolError",
    "ToolInvocationError",
    "ToolRegistry",
    "ToolResult",
    "ToolRun",
    "ToolSpec",
]
