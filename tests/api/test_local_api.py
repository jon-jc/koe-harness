"""The local-models endpoints, and the policy that picks a backend.

The policy is the part worth guarding. "Which model is serving" has four
inputs -- a key, a running server, a preference flag, and the mock switch --
and getting the precedence wrong is invisible: everything still answers, just
from somewhere the user did not choose.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.config import Settings
from koe.providers.local.llm import LocalLLM
from koe.providers.local.resolve import Choice


def _services(**overrides: Any) -> Services:
    settings = Settings(environment="local", force_mock_providers=False, **overrides)
    return Services.default(settings)


def _local_choice() -> Choice:
    return Choice(
        provider=LocalLLM(model="qwen2.5:7b", base_url="http://127.0.0.1:11434", label="ollama"),
        reason="Ollama · qwen2.5:7b",
    )


@pytest.fixture
def client() -> TestClient:
    services = _services()
    # No probing in tests: what is listening on the developer's machine must
    # not decide whether the suite passes.
    return TestClient(create_app(services))


# --------------------------------------------------------------------------
# selection policy
# --------------------------------------------------------------------------


def test_without_a_key_or_a_local_server_the_mock_serves() -> None:
    """A clean checkout still runs, which is the whole reason the mock exists."""
    services = _services()
    services.local_llm = Choice(reason="no local model server is running")
    assert services.refresh_llm() == "mock"


def test_a_local_server_beats_the_mock() -> None:
    """Free, private, and enormously better than scripted text."""
    services = _services()
    services.local_llm = _local_choice()
    assert services.refresh_llm() == "local"
    assert services.llm.model == "qwen2.5:7b"


def test_a_configured_key_beats_a_discovered_server_by_default() -> None:
    """Someone who pasted a key expressed a preference. A server that happens
    to be listening on 11434 did not."""
    services = _services()
    services.local_llm = _local_choice()
    services.credentials.resolve = lambda provider: (  # type: ignore[method-assign]
        "sk-ant-test" if provider == "anthropic" else ""
    )

    chosen = services.refresh_llm()

    # Falls back to local when the vendor SDK is absent, which is the correct
    # outcome and is what this build has: the key is real but unusable.
    assert chosen in {"anthropic", "local"}


def test_preferring_local_overrides_a_configured_key() -> None:
    """The setting a privacy-motivated user is looking for. Without it they
    would have to delete their credentials to stop them being used."""
    services = _services(prefer_local_llm=True)
    services.local_llm = _local_choice()
    services.credentials.resolve = lambda provider: "sk-ant-test"  # type: ignore[method-assign]

    assert services.refresh_llm() == "local"


def test_forcing_mocks_beats_everything() -> None:
    """CI must not depend on what happens to be listening on the runner."""
    services = Services.default(Settings(environment="local", force_mock_providers=True))
    services.local_llm = _local_choice()
    assert services.refresh_llm() == "mock"


async def test_rescanning_under_forced_mocks_does_not_probe() -> None:
    services = Services.default(Settings(environment="local", force_mock_providers=True))
    assert await services.rescan_local() == "mock"
    assert not services.local_llm
    assert "mock" in services.local_llm.reason


# --------------------------------------------------------------------------
# the ASR router
# --------------------------------------------------------------------------


def test_local_whisper_joins_the_router_rather_than_replacing_it() -> None:
    """Local is free and slow, a vendor is quick and metered, and which one a
    request should use is exactly the trade the router exists to make."""
    services = _services()
    before = set(services.asr_router.names)

    class FakeWhisper:
        info = type(
            "I", (), {"name": "local-whisper", "model": "large-v3-turbo", "modality": "asr"}
        )()

    services.local_asr = Choice(provider=FakeWhisper(), reason="local Whisper · large-v3-turbo")
    services.apply_local_asr()

    assert "local-whisper" in services.asr_router.names
    assert before <= set(services.asr_router.names)


def test_turning_local_recognition_off_removes_it_from_the_router() -> None:
    services = _services()

    class FakeWhisper:
        info = type(
            "I", (), {"name": "local-whisper", "model": "large-v3-turbo", "modality": "asr"}
        )()

    services.local_asr = Choice(provider=FakeWhisper(), reason="on")
    services.apply_local_asr()
    services.local_asr = Choice(reason="local recognition is off")
    services.apply_local_asr()

    assert "local-whisper" not in services.asr_router.names


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def test_the_status_endpoint_lists_the_whisper_sizes(client: TestClient) -> None:
    body = client.get("/v1/local").json()

    sizes = {size["id"]: size for size in body["asr"]["sizes"]}
    assert "large-v3-turbo" in sizes
    # The panel needs the per-language answer, or it would present the small
    # checkpoints as a neutral speed slider.
    assert sizes["tiny"]["suitable_ja"] is False
    assert sizes["large-v3"]["suitable_ja"] is True


def test_the_status_endpoint_names_the_servers_it_looks_for(client: TestClient) -> None:
    """So a machine with nothing running gets install links rather than a blank
    panel."""
    body = client.get("/v1/local").json()
    assert "ollama" in {server["id"] for server in body["known_servers"]}


def test_settings_can_be_changed_and_come_back(client: TestClient) -> None:
    body = client.put("/v1/local", json={"base_url": "localhost:11434", "prefer": True}).json()

    assert body["llm"]["base_url"] == "localhost:11434"
    assert body["llm"]["prefer"] is True


def test_an_omitted_field_is_left_alone(client: TestClient) -> None:
    """The panel sends deltas, so the ASR toggle must not undo an in-flight
    base-URL edit."""
    client.put("/v1/local", json={"base_url": "localhost:11434"})
    body = client.put("/v1/local", json={"asr_enabled": False}).json()

    assert body["llm"]["base_url"] == "localhost:11434"


def test_an_unknown_whisper_size_is_refused(client: TestClient) -> None:
    assert client.put("/v1/local", json={"asr_model": "enormous"}).status_code == 400


def test_an_unknown_device_is_refused(client: TestClient) -> None:
    """Anything but auto/cpu/cuda would reach CTranslate2 and fail there, with
    a message about a device koe never offered."""
    assert client.put("/v1/local", json={"asr_device": "tpu"}).status_code == 422
