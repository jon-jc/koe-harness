"""koe (声) -- a plugin harness for bilingual Japanese/English voice AI.

koe orchestrates the several models a real voice product needs -- streaming
ASR, speaker diarization, and an LLM writing 議事録 (meeting minutes) -- behind
one composable plugin kernel, and holds them to explicit latency, cost and
quality budgets.

Quick start::

    from koe import Context
    from koe.providers.asr import MockASR

    ctx = Context()
    ctx.provide("asr", MockASR())
    ctx.plugin(my_plugin)

See ``docs/architecture.md`` for the full design.
"""

from koe.kernel import Context, KoeError, Scope, plugin

__version__ = "0.1.0"

__all__ = ["Context", "KoeError", "Scope", "__version__", "plugin"]
