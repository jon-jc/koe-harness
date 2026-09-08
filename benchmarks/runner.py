"""A small benchmark harness.

Three things it does that a bare stopwatch does not, each because the number is
otherwise misleading:

**Warmup.** The first call pays for imports, MeCab dictionary loading, and JIT
of nothing in particular — measuring it reports startup cost as steady-state
throughput.

**Percentiles, not means.** Same argument as the metrics layer: a mean is
dominated by the fast majority and hides the tail, and on a realtime path the
tail is the thing that breaks a latency budget.

**Real-time factor where it applies.** For anything on the audio path, "12 ms
per call" is meaningless without knowing how much audio the call covered. RTF —
processing seconds per audio second — is the number that says whether a stream
keeps up, and it is the number that has to stay well under 1.0 for concurrency
to be possible at all.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Result:
    """One benchmark's measurements."""

    name: str
    group: str
    samples: list[float] = field(default_factory=list, repr=False)
    #: Units of work per call (characters, utterances, audio seconds...).
    units_per_call: float = 1.0
    unit_label: str = "op"
    #: Set for audio-path benchmarks, where RTF is the meaningful figure.
    audio_seconds_per_call: float = 0.0
    peak_bytes: int = 0

    @property
    def p50_ms(self) -> float:
        return self._percentile(0.50) * 1000

    @property
    def p95_ms(self) -> float:
        return self._percentile(0.95) * 1000

    @property
    def p99_ms(self) -> float:
        return self._percentile(0.99) * 1000

    @property
    def throughput(self) -> float:
        """Units of work per second, at the median."""
        median = self._percentile(0.50)
        return self.units_per_call / median if median > 0 else 0.0

    @property
    def rtf(self) -> float:
        """Processing seconds per audio second. Below 1.0 keeps up."""
        if self.audio_seconds_per_call <= 0:
            return 0.0
        return self._percentile(0.50) / self.audio_seconds_per_call

    @property
    def max_concurrent(self) -> int:
        """How many streams one core could sustain at this RTF.

        A ceiling, not a capacity plan: it ignores GIL contention, network
        waits and every other real cost. Useful for spotting a component that
        cannot support concurrency at all.
        """
        if self.rtf <= 0:
            return 0
        return int(1.0 / self.rtf)

    def _percentile(self, fraction: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[int(position)]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Bench:
    """Collects benchmarks and runs them."""

    def __init__(self) -> None:
        self._cases: list[tuple[str, str, Callable[[], Any], dict[str, Any]]] = []

    def case(
        self,
        name: str,
        *,
        group: str,
        units_per_call: float = 1.0,
        unit_label: str = "op",
        audio_seconds_per_call: float = 0.0,
        repeats: int = 200,
        warmup: int = 20,
    ) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
        """Register a benchmark."""

        def register(fn: Callable[[], Any]) -> Callable[[], Any]:
            self._cases.append(
                (
                    name,
                    group,
                    fn,
                    {
                        "units_per_call": units_per_call,
                        "unit_label": unit_label,
                        "audio_seconds_per_call": audio_seconds_per_call,
                        "repeats": repeats,
                        "warmup": warmup,
                    },
                )
            )
            return fn

        return register

    def run(self, *, only: str | None = None) -> list[Result]:
        results: list[Result] = []
        for name, group, fn, options in self._cases:
            if only and only not in group and only not in name:
                continue
            results.append(_measure(name, group, fn, **options))
        return results

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter((name, group) for name, group, _, _ in self._cases)


def _measure(
    name: str,
    group: str,
    fn: Callable[[], Any],
    *,
    units_per_call: float,
    unit_label: str,
    audio_seconds_per_call: float,
    repeats: int,
    warmup: int,
) -> Result:
    for _ in range(warmup):
        fn()

    # GC during a timed run shows up as a tail that belongs to the allocator,
    # not to the code under test. Disabled for the measurement, collected once
    # before it so the heap starts in a comparable state.
    gc.collect()
    gc.disable()
    samples: list[float] = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - started)
    finally:
        gc.enable()

    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Result(
        name=name,
        group=group,
        samples=samples,
        units_per_call=units_per_call,
        unit_label=unit_label,
        audio_seconds_per_call=audio_seconds_per_call,
        peak_bytes=peak,
    )


def _fmt_rtf(rtf: float) -> str:
    """Format a real-time factor without flattening the good ones to zero.

    The interesting RTFs here are 1e-4 and below; four decimal places renders
    every one of them as "0.0000", which reads as missing data rather than as
    "four orders of magnitude of headroom".
    """
    if rtf <= 0:
        return "—"
    if rtf < 0.001:
        return f"{rtf:.1e}"
    return f"{rtf:.4f}"


def render(results: list[Result]) -> str:
    """Format results as a table, grouped."""
    if not results:
        return "no benchmarks matched"

    lines: list[str] = []
    width = max(len(r.name) for r in results) + 2

    for group in dict.fromkeys(r.group for r in results):
        lines.append("")
        lines.append(group)
        lines.append("-" * 78)
        lines.append(f"{'benchmark':<{width}} {'p50':>9} {'p95':>9} {'throughput':>18} {'RTF':>8}")
        for result in (r for r in results if r.group == group):
            rtf = _fmt_rtf(result.rtf)
            throughput = f"{result.throughput:,.0f} {result.unit_label}/s"
            lines.append(
                f"{result.name:<{width}} "
                f"{result.p50_ms:>7.3f}ms {result.p95_ms:>7.3f}ms "
                f"{throughput:>18} {rtf:>8}"
            )
    return "\n".join(lines)


def render_markdown(results: list[Result]) -> str:
    """Format results as Markdown, for pasting into BENCHMARK.md."""
    lines: list[str] = []
    for group in dict.fromkeys(r.group for r in results):
        lines.append(f"\n### {group}\n")
        lines.append("| benchmark | p50 | p95 | throughput | RTF |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in (x for x in results if x.group == group):
            rtf = f"`{_fmt_rtf(r.rtf)}`" if r.rtf else "—"
            lines.append(
                f"| {r.name} | {r.p50_ms:.3f} ms | {r.p95_ms:.3f} ms | "
                f"{r.throughput:,.0f} {r.unit_label}/s | {rtf} |"
            )
    return "\n".join(lines)


def summarize(results: list[Result]) -> str:
    audio = [r for r in results if r.rtf > 0]
    if not audio:
        return ""
    worst = max(audio, key=lambda r: r.rtf)
    return (
        f"\nslowest audio-path stage: {worst.name} at RTF {worst.rtf:.4f} "
        f"(~{worst.max_concurrent} concurrent streams per core, ceiling)"
    )


#: Median of a list, used by a couple of benches.
median = statistics.median
