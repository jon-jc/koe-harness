"""Metrics and structured logging.

Two decisions here, both about what is actually usable at 3am.

**Latency is recorded as a histogram, never a mean.** The mean latency of a
voice pipeline is a number that describes nobody's experience: it is dominated
by the many fast requests and hides the slow tail entirely. A p95 that doubled
is an incident; a mean that moved 4% is noise that looks identical. So the
percentiles are what gets exported.

**Logs are structured and carry a session id.** A voice session produces
interleaved output from several concurrent tasks -- endpointing, partial
recognition, the final pass, the websocket pump. Grepping that back into one
story from free-text lines is guesswork; a `session_id` field makes it a
filter.

The exporter emits **CloudWatch EMF** (Embedded Metric Format), which is a
JSON log line CloudWatch parses into metrics. That choice means metrics need no
agent, no sidecar, and no extra network path -- on ECS/Fargate stdout already
goes to CloudWatch Logs, so emitting a specific JSON shape is the entire
integration. Somewhere else, the same records go to whatever reads stdout.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Histogram:
    """Latency samples, summarized by percentile.

    Stores raw samples up to `capacity` and then reservoir-samples, so memory
    is bounded on a long-running process while the percentile estimate stays
    representative.
    """

    name: str
    capacity: int = 4096
    _samples: list[float] = field(default_factory=list, repr=False)
    _count: int = 0

    def observe(self, value: float) -> None:
        self._count += 1
        if len(self._samples) < self.capacity:
            self._samples.append(value)
            return
        # Reservoir sampling: every observation keeps an equal chance of being
        # retained, so the distribution does not drift toward the first N.
        index = int(_deterministic_slot(self._count) * self._count)
        if index < self.capacity:
            self._samples[index] = value

    @property
    def count(self) -> int:
        return self._count

    def percentile(self, fraction: float) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[int(position)]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    def snapshot(self) -> dict[str, float]:
        return {
            "count": float(self._count),
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "max": max(self._samples) if self._samples else 0.0,
        }


def _deterministic_slot(seed: int) -> float:
    """A cheap, reproducible [0,1) draw.

    Deterministic on purpose: two processes replaying the same event sequence
    produce the same summary, which makes a metrics discrepancy debuggable
    rather than a coin flip.
    """
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return x / 0x7FFFFFFF


class Metrics:
    """Counters and histograms, exported as CloudWatch EMF."""

    def __init__(self, namespace: str = "koe") -> None:
        self.namespace = namespace
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = Histogram(name=name)
                self._histograms[name] = histogram
            histogram.observe(value)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Time a block and record it as a histogram sample."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {n: h.snapshot() for n, h in self._histograms.items()},
            }

    def to_emf(self, dimensions: dict[str, str] | None = None) -> dict[str, Any]:
        """Render as a CloudWatch Embedded Metric Format record.

        One JSON line on stdout becomes CloudWatch metrics with no agent and no
        extra network path, because on Fargate stdout already goes to
        CloudWatch Logs.
        """
        dims = dimensions or {}
        snapshot = self.snapshot()

        definitions: list[dict[str, str]] = []
        values: dict[str, Any] = {}

        for name, value in snapshot["counters"].items():
            definitions.append({"Name": name, "Unit": "Count"})
            values[name] = value

        for name, stats in snapshot["histograms"].items():
            for suffix in ("p50", "p95", "p99"):
                metric = f"{name}_{suffix}"
                definitions.append({"Name": metric, "Unit": "Milliseconds"})
                values[metric] = stats[suffix]

        return {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self.namespace,
                        "Dimensions": [list(dims)] if dims else [[]],
                        "Metrics": definitions,
                    }
                ],
            },
            **dims,
            **values,
        }

    def emit(self, dimensions: dict[str, str] | None = None) -> None:
        """Write one EMF record to stdout."""
        print(json.dumps(self.to_emf(dimensions)), file=sys.stdout, flush=True)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# --------------------------------------------------------------------------
# structured logging
# --------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON.

    Extra fields set on the record (``logger.info(..., extra={"session_id": x})``)
    are merged in, which is what makes a session's interleaved output filterable
    rather than something you reassemble by eye.
    """

    _RESERVED = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "taskName", "message", "asctime",
        }
    )  # fmt: skip

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, structured: bool = True) -> None:
    """Install the root logging configuration.

    Structured by default because this runs in a container, where logs are
    read by a machine before a human. Pass ``structured=False`` for a readable
    local console.
    """
    handler = logging.StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty at INFO and say nothing a request log does not.
    for noisy in ("uvicorn.access", "multipart", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


#: Process-wide default, so callers do not have to thread one through.
METRICS = Metrics()
