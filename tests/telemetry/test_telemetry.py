"""Cost ledger, metrics, and structured logging."""

from __future__ import annotations

import json
import logging

import pytest

from koe.providers.base import Usage
from koe.telemetry.ledger import BudgetExceeded, CostLedger
from koe.telemetry.metrics import Histogram, JSONFormatter, Metrics


def usage(cost: float, *, audio: float = 60.0, provider: str = "p", model: str = "m") -> Usage:
    return Usage(provider=provider, model=model, audio_seconds=audio, cost_usd=cost)


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def test_spend_is_reported_per_audio_hour() -> None:
    """Total spend says nothing without knowing how much audio produced it."""
    ledger = CostLedger()
    ledger.record(usage(0.10, audio=1800.0))  # 30 minutes

    assert ledger.total.cost_usd == pytest.approx(0.10)
    assert ledger.total.cost_per_audio_hour == pytest.approx(0.20)


def test_spend_is_broken_down_by_model() -> None:
    """'Which of these is expensive' is unanswerable from one invoice number."""
    ledger = CostLedger()
    ledger.record(usage(0.02, provider="whisper", model="local"))
    ledger.record(usage(0.30, provider="anthropic", model="claude-opus-5"))
    ledger.record(usage(0.01, provider="whisper", model="local"))

    breakdown = ledger.by_model()

    assert breakdown["whisper/local"].calls == 2
    assert breakdown["whisper/local"].cost_usd == pytest.approx(0.03)
    assert breakdown["anthropic/claude-opus-5"].cost_usd == pytest.approx(0.30)


def test_spend_is_tracked_per_session_and_tenant() -> None:
    ledger = CostLedger()
    ledger.record(usage(0.05), session_id="s1", tenant="acme")
    ledger.record(usage(0.07), session_id="s2", tenant="acme")
    ledger.record(usage(0.01), session_id="s3", tenant="other")

    assert ledger.by_session("s1").cost_usd == pytest.approx(0.05)
    assert ledger.by_tenant("acme").cost_usd == pytest.approx(0.12)
    assert ledger.by_tenant("other").cost_usd == pytest.approx(0.01)


def test_a_runaway_session_is_stopped() -> None:
    """The failure that turns a rounding error into a real bill."""
    ledger = CostLedger(session_limit_usd=0.10)
    ledger.record(usage(0.06), session_id="runaway")

    with pytest.raises(BudgetExceeded) as excinfo:
        ledger.record(usage(0.09), session_id="runaway")

    assert excinfo.value.limit == pytest.approx(0.10)
    assert excinfo.value.spent > 0.10


def test_the_offending_call_is_still_recorded() -> None:
    """Money already spent does not un-spend itself by breaking a limit."""
    ledger = CostLedger(session_limit_usd=0.05)
    with pytest.raises(BudgetExceeded):
        ledger.record(usage(0.09), session_id="s")

    assert ledger.by_session("s").cost_usd == pytest.approx(0.09)
    assert ledger.total.calls == 1


def test_tenant_budgets_are_enforced_across_sessions() -> None:
    ledger = CostLedger(tenant_limit_usd=0.10)
    ledger.record(usage(0.06), session_id="a", tenant="acme")

    with pytest.raises(BudgetExceeded, match="tenant"):
        ledger.record(usage(0.06), session_id="b", tenant="acme")


def test_the_entry_buffer_is_bounded() -> None:
    """An unbounded list is a slow memory leak in a long-lived server."""
    ledger = CostLedger(max_entries=10)
    for _ in range(50):
        ledger.record(usage(0.001))

    assert len(ledger.recent(100)) <= 10
    # totals remain authoritative even though detail was dropped
    assert ledger.total.calls == 50
    assert ledger.total.cost_usd == pytest.approx(0.05)


def test_report_orders_by_cost() -> None:
    ledger = CostLedger()
    ledger.record(usage(0.01, provider="cheap", model="a"))
    ledger.record(usage(0.50, provider="pricey", model="b"))

    report = ledger.report()

    assert report.index("pricey/b") < report.index("cheap/a")
    assert "audio-hour" in report


def test_reset_clears_everything() -> None:
    ledger = CostLedger()
    ledger.record(usage(0.1), session_id="s")
    ledger.reset()
    assert ledger.total.calls == 0
    assert ledger.by_session("s").cost_usd == 0.0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_histogram_reports_percentiles_not_a_mean() -> None:
    """A mean latency describes nobody's experience and hides the tail."""
    histogram = Histogram(name="latency")
    for value in [10.0] * 95 + [500.0] * 5:
        histogram.observe(value)

    stats = histogram.snapshot()

    assert stats["p50"] == pytest.approx(10.0)
    assert stats["p95"] > 10.0
    assert stats["max"] == pytest.approx(500.0)
    assert stats["count"] == 100


def test_histogram_memory_is_bounded() -> None:
    histogram = Histogram(name="latency", capacity=100)
    for i in range(10_000):
        histogram.observe(float(i))

    assert len(histogram._samples) == 100
    assert histogram.count == 10_000


def test_histogram_summary_is_reproducible() -> None:
    """Two processes replaying the same events must summarize identically."""

    def build() -> dict[str, float]:
        histogram = Histogram(name="x", capacity=50)
        for i in range(1_000):
            histogram.observe(float(i % 37))
        return histogram.snapshot()

    assert build() == build()


def test_empty_histogram() -> None:
    assert Histogram(name="x").snapshot()["p95"] == 0.0


def test_counters_and_timers() -> None:
    metrics = Metrics()
    metrics.increment("sessions")
    metrics.increment("sessions")
    metrics.increment("frames", 128)
    with metrics.timer("asr.latency"):
        pass

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["sessions"] == 2
    assert snapshot["counters"]["frames"] == 128
    assert snapshot["histograms"]["asr.latency"]["count"] == 1


def test_emf_export_shape() -> None:
    """CloudWatch parses this straight off stdout -- no agent, no sidecar."""
    metrics = Metrics(namespace="koe-test")
    metrics.increment("sessions", 3)
    metrics.observe("asr.latency", 120.0)

    record = metrics.to_emf({"Service": "koe", "Stage": "prod"})

    assert record["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "koe-test"
    assert record["Service"] == "koe"
    assert record["sessions"] == 3
    assert "asr.latency_p95" in record
    names = {m["Name"] for m in record["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert "sessions" in names
    assert "asr.latency_p50" in names
    json.dumps(record)  # must be serializable as one log line


def test_emf_without_dimensions() -> None:
    assert Metrics().to_emf()["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [[]]


# --------------------------------------------------------------------------
# structured logging
# --------------------------------------------------------------------------


def test_log_records_are_single_line_json() -> None:
    record = logging.LogRecord(
        name="koe.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="recognized %s segments",
        args=(3,),
        exc_info=None,
    )
    payload = json.loads(JSONFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "recognized 3 segments"
    assert payload["logger"] == "koe.test"


def test_extra_fields_are_merged_so_a_session_is_filterable() -> None:
    """A voice session interleaves output from several concurrent tasks."""
    record = logging.LogRecord(
        name="koe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="endpoint",
        args=(),
        exc_info=None,
    )
    record.session_id = "abc123"  # type: ignore[attr-defined]
    record.provider = "whisper"  # type: ignore[attr-defined]

    payload = json.loads(JSONFormatter().format(record))

    assert payload["session_id"] == "abc123"
    assert payload["provider"] == "whisper"


def test_exceptions_are_captured() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="koe",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(JSONFormatter().format(record))
    assert "ValueError" in payload["exception"]


def test_japanese_survives_serialization() -> None:
    record = logging.LogRecord(
        name="koe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="議事録を作成しました",
        args=(),
        exc_info=None,
    )
    payload = json.loads(JSONFormatter().format(record))
    assert payload["message"] == "議事録を作成しました"
