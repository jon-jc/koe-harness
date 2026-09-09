"""Example koe plugin: who talked, and for how long.

Copy this file into the plugins directory to try it:

    %LOCALAPPDATA%/koe/plugins         (Windows)
    ~/.local/share/koe/plugins          (macOS, Linux)

It is deliberately small but complete: metadata, a dependency declaration, a
tool with a model-facing description, argument validation that the model can
recover from, and a disposer on the scope so disabling the plugin actually
removes the tool.

See docs/plugins.md for the full contract.
"""

from __future__ import annotations

from typing import Any

KOE_PLUGIN = {
    "name": "speaker-stats",
    "description": "Adds a tool reporting who spoke and for how long.",
    "version": "1.0.0",
    # Declares that this plugin cannot run before the tool registry exists.
    # The kernel activates it the moment that service appears, and rebuilds it
    # if the service is ever replaced.
    "inject": ["tools"],
}


def apply(ctx: Any, config: Any = None) -> None:
    from koe.tools import ToolInvocationError, ToolSpec

    registry = ctx.get("tools")

    async def speaker_stats(args: dict[str, Any], run: Any) -> Any:
        segments = ctx.get("last_segments") or []
        if not segments:
            return "No meeting has been transcribed in this session yet."

        minimum = args.get("min_seconds", 0)
        try:
            floor = float(minimum)
        except (TypeError, ValueError):
            # A wrong argument is a normal outcome the model can fix, so it is
            # reported as one rather than crashing the tool.
            raise ToolInvocationError("min_seconds must be a number") from None

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for segment in segments:
            speaker = segment.get("speaker") or "unknown"
            duration = float(segment.get("end", 0)) - float(segment.get("start", 0))
            totals[speaker] = totals.get(speaker, 0.0) + duration
            counts[speaker] = counts.get(speaker, 0) + 1

        spoken = sum(totals.values()) or 1.0
        rows = [
            f"{speaker}: {seconds:.1f}s ({seconds / spoken:.0%}, "
            f"{counts[speaker]} turn{'' if counts[speaker] == 1 else 's'})"
            for speaker, seconds in sorted(totals.items(), key=lambda kv: -kv[1])
            if seconds >= floor
        ]
        return "\n".join(rows) if rows else "No speaker met that threshold."

    spec = ToolSpec(
        name="speaker_stats",
        description=(
            "Report how long each participant spoke in the current meeting, as "
            "seconds, share of the total, and number of turns. Use it to answer "
            "questions about participation or balance; use current_transcript "
            "when you need what was actually said."
        ),
        parameters={
            "type": "object",
            "properties": {
                "min_seconds": {
                    "type": "number",
                    "description": "Omit speakers below this many seconds.",
                }
            },
        },
        execute=speaker_stats,
        source="speaker-stats",
    )

    # The disposer goes on the scope, so disabling this plugin removes the tool
    # rather than leaving it pointing at unloaded code.
    ctx.scope.collect("tool:speaker_stats", registry.register(spec))
