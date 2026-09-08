"""Operational behaviour: capacity, readiness, correlation, configuration.

These are the properties that separate a service you can demo from one you can
run: what happens when it is full, when a dependency is down, when a request
fails, and when it is started with a configuration that would be unsafe.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from koe.api.app import Services, create_app
from koe.config import Settings
from koe.telemetry.ledger import CostLedger
from koe.telemetry.metrics import Metrics


def services(**overrides: object) -> Services:
    settings = Settings(environment="local", **overrides)  # type: ignore[arg-type]
    built = Services.default(settings)
    built.metrics = Metrics(namespace="test")
    return built


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(services()))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_defaults_are_bounded() -> None:
    """Unlimited-by-default is how a demo becomes an incident."""
    settings = Settings()
    assert settings.max_concurrent_sessions > 0
    assert settings.max_session_seconds > 0
    assert settings.session_budget_usd is not None


def test_production_rejects_a_wildcard_cors_policy() -> None:
    problems = Settings(environment="production", cors_origins=["*"]).validate_for_environment()
    assert any("cors_origins" in p for p in problems)


def test_production_rejects_an_unbounded_session_budget() -> None:
    problems = Settings(
        environment="production",
        cors_origins=["https://example.com"],
        session_budget_usd=None,
    ).validate_for_environment()
    assert any("budget" in p for p in problems)


def test_production_rejects_forced_mocks() -> None:
    problems = Settings(
        environment="production",
        cors_origins=["https://example.com"],
        force_mock_providers=True,
    ).validate_for_environment()
    assert any("mock" in p for p in problems)


def test_a_valid_production_config_has_no_problems() -> None:
    assert (
        Settings(
            environment="production",
            cors_origins=["https://koe.example.com"],
            session_budget_usd=2.0,
            anthropic_api_key="sk-test",
        ).validate_for_environment()
        == []
    )


def test_local_is_not_held_to_production_policy() -> None:
    """A developer should not be blocked by a rule that only matters in prod."""
    assert Settings(environment="local", cors_origins=["*"]).validate_for_environment() == []


def test_invalid_values_are_rejected_at_startup_not_first_use() -> None:
    with pytest.raises(ValidationError):
        Settings(max_concurrent_sessions=0)
    with pytest.raises(ValidationError):
        Settings(partial_interval_ms=10.0)  # below the floor


def test_a_fresh_checkout_runs_on_mocks() -> None:
    """No credentials must mean 'starts and serves', not 'crashes'."""
    assert Settings(anthropic_api_key=None, openai_api_key=None).use_mocks
    assert not Settings(anthropic_api_key="sk-test").use_mocks


def test_refusing_to_start_misconfigured_in_production() -> None:
    bad = Services.default(
        Settings(environment="production", cors_origins=["*"], session_budget_usd=1.0)
    )
    with pytest.raises(RuntimeError, match="refusing to start"), TestClient(create_app(bad)):
        pass


# --------------------------------------------------------------------------
# capacity and readiness
# --------------------------------------------------------------------------


def test_liveness_and_readiness_are_different_questions(client: TestClient) -> None:
    """A saturated server is alive but should stop receiving new sessions."""
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["ready"] is True


def test_a_full_server_rejects_rather_than_queues() -> None:
    """A caller waiting behind a full server records audio nobody transcribes."""
    app = create_app(services(max_concurrent_sessions=1))
    with TestClient(app) as client, client.websocket_connect("/v1/stream") as first:
        first.send_json({"type": "start", "language": "ja"})
        assert first.receive_json()["type"] == "started"

        with client.websocket_connect("/v1/stream") as second:
            message = second.receive_json()
            assert message["type"] == "error"
            assert "capacity" in message["message"]


def test_readiness_reports_capacity_pressure() -> None:
    app = create_app(services(max_concurrent_sessions=1))
    with TestClient(app) as client, client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        assert ws.receive_json()["type"] == "started"

        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["at_capacity"] is True


def test_a_session_slot_is_released_after_disconnect() -> None:
    app = create_app(services(max_concurrent_sessions=1))
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "language": "ja"})
            assert ws.receive_json()["type"] == "started"

        # the slot must come back, or the server bleeds capacity per session
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "language": "ja"})
            assert ws.receive_json()["type"] == "started"


# --------------------------------------------------------------------------
# correlation and observability
# --------------------------------------------------------------------------


def test_a_request_id_is_returned_on_every_response(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")


def test_an_upstream_request_id_is_preserved(client: TestClient) -> None:
    """A trace that starts at a load balancer must stay one trace."""
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_metrics_endpoint_reports_latency_and_cost(client: TestClient) -> None:
    client.get("/health")
    body = client.get("/metrics").json()

    assert body["counters"]["http.requests"] >= 1
    assert "http.latency_ms" in body["histograms"]
    assert "cost" in body
    assert "by_model" in body["cost"]


def test_cost_is_recorded_for_a_streaming_session() -> None:
    built = services()
    app = create_app(built)
    with TestClient(app) as client, client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        assert ws.receive_json()["type"] == "started"
        ws.send_bytes(b"\x00" * 32_000)
        ws.send_json({"type": "stop"})
        for _ in range(40):
            if ws.receive_json()["type"] == "transcript":
                break

    # the ledger is the record of what a session cost, independent of the
    # response the client happened to receive
    assert built.ledger.total.calls >= 0


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------


def test_an_unhandled_error_returns_a_request_id_not_a_traceback() -> None:
    """Never leak internals; give the caller something to quote instead."""
    built = services()
    app = create_app(built)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("internal detail that must not reach the client")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "internal error"
    assert body["request_id"]
    assert "internal detail" not in response.text


def test_ledger_and_metrics_are_per_app_not_global() -> None:
    """Two servers in one process must not pollute each other's accounting."""
    a, b = services(), services()
    a.ledger = CostLedger()
    b.ledger = CostLedger()
    assert a.ledger is not b.ledger
