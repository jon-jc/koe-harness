"""The credential endpoints.

The property worth defending at this layer is that a key goes in and never
comes back out — not in a response, not in an error, not in a log.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.config import Settings
from koe.providers.credentials import (
    KNOWN_PROVIDERS,
    PROVIDERS_BY_ID,
    CredentialStore,
)

ANTHROPIC_KEY = "sk-ant-api03-" + "T3stK3y" * 8


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for spec in KNOWN_PROVIDERS:
        monkeypatch.delenv(spec.env_var, raising=False)
    services = Services.default(Settings(environment="local"))
    services.credentials = CredentialStore(tmp_path / "credentials.json")
    services.refresh_llm()
    return TestClient(create_app(services))


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------


def test_listing_shows_every_known_provider(client: TestClient) -> None:
    body = client.get("/v1/credentials").json()
    listed = {p["provider"] for p in body["providers"]}
    assert listed == {spec.id for spec in KNOWN_PROVIDERS}
    assert body["active_llm"] == "mock-llm"


def test_an_unconfigured_provider_reports_no_fingerprint(client: TestClient) -> None:
    body = client.get("/v1/credentials").json()
    anthropic = next(p for p in body["providers"] if p["provider"] == "anthropic")
    assert anthropic["configured"] is False
    assert anthropic["fingerprint"] == ""
    assert anthropic["docs_url"].startswith("https://")


# --------------------------------------------------------------------------
# storing
# --------------------------------------------------------------------------


def test_a_stored_key_is_never_returned(client: TestClient) -> None:
    """The whole point: it goes in, only a fingerprint comes out."""
    response = client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY})
    assert response.status_code == 200
    assert ANTHROPIC_KEY not in response.text

    listing = client.get("/v1/credentials")
    assert ANTHROPIC_KEY not in listing.text

    anthropic = next(p for p in listing.json()["providers"] if p["provider"] == "anthropic")
    assert anthropic["configured"] is True
    assert anthropic["fingerprint"].endswith(ANTHROPIC_KEY[-4:])


ANTHROPIC_SDK = PROVIDERS_BY_ID["anthropic"].available


@pytest.mark.skipif(not ANTHROPIC_SDK, reason="the anthropic SDK is not installed")
def test_saving_a_key_switches_the_live_provider(client: TestClient) -> None:
    """A pasted key takes effect on the next request, not after a restart."""
    assert client.get("/v1/credentials").json()["active_llm"] == "mock-llm"

    body = client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY}).json()

    assert body["active_llm"] == "anthropic"
    assert client.get("/v1/credentials").json()["active_llm"] == "anthropic"


@pytest.mark.skipif(ANTHROPIC_SDK, reason="the anthropic SDK is installed")
def test_a_key_without_its_sdk_stays_on_the_mock(client: TestClient) -> None:
    """A backend this build cannot import must not be advertised as active.

    Otherwise the settings panel says "anthropic", the user believes their key
    took effect, and the first 議事録 request fails on an ImportError.
    """
    body = client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY}).json()

    assert body["active_llm"] == "mock"
    assert body["provider"]["configured"] is True
    assert body["provider"]["available"] is False


def test_removing_a_key_falls_back(client: TestClient) -> None:
    client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY})
    body = client.delete("/v1/credentials/anthropic").json()
    assert body["active_llm"] == "mock"
    assert body["provider"]["configured"] is False


def test_a_malformed_key_is_rejected_with_a_reason(client: TestClient) -> None:
    response = client.put(
        "/v1/credentials/anthropic", json={"key": "not-a-key-but-long-enough-to-pass"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]
    # The code, not the prose, is what the bilingual client keys off.
    assert body["code"] == "bad_prefix"
    assert body["context"]["prefix"] == "sk-ant-"


def test_a_too_short_key_is_rejected_by_the_schema(client: TestClient) -> None:
    assert client.put("/v1/credentials/anthropic", json={"key": "abc"}).status_code == 422


def test_an_unknown_provider_is_a_404(client: TestClient) -> None:
    response = client.put("/v1/credentials/nonesuch", json={"key": ANTHROPIC_KEY})
    assert response.status_code == 404


def test_an_environment_credential_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writing a value the resolver will never read is worse than refusing."""
    monkeypatch.setenv("KOE_ANTHROPIC_API_KEY", "sk-ant-from-the-environment-xxxx")
    services = Services.default(Settings(environment="local"))
    services.credentials = CredentialStore(tmp_path / "credentials.json")

    with TestClient(create_app(services)) as client:
        response = client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY})

    assert response.status_code == 409
    body = response.json()
    assert "KOE_ANTHROPIC_API_KEY" in body["detail"]
    assert body["code"] == "env_managed"
    assert body["context"]["env"] == "KOE_ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def test_verifying_without_a_key_reports_unknown(client: TestClient) -> None:
    body = client.post("/v1/credentials/anthropic/verify").json()
    assert body["provider"]["status"] == "unknown"
    assert "no key" in body["provider"]["detail"]


def test_verifying_an_unknown_provider_is_a_404(client: TestClient) -> None:
    assert client.post("/v1/credentials/nonesuch/verify").status_code == 404


def test_a_bad_key_verifies_as_invalid_or_error(client: TestClient) -> None:
    """Live call. Either outcome is acceptable; 'valid' would not be.

    invalid  = the vendor rejected it (the normal result)
    error    = no network, or the client library is absent in this build
    """
    client.put("/v1/credentials/anthropic", json={"key": ANTHROPIC_KEY})
    body = client.post("/v1/credentials/anthropic/verify").json()

    assert body["provider"]["status"] in {"invalid", "error"}
    assert body["provider"]["detail"]
    # Whatever happened, the key must not appear in the explanation.
    assert ANTHROPIC_KEY not in body["provider"]["detail"]


def test_forced_mock_keeps_the_mock_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator pinning mocks must not be overridden by a stored key."""
    for spec in KNOWN_PROVIDERS:
        monkeypatch.delenv(spec.env_var, raising=False)
    services = Services.default(Settings(environment="local", force_mock_providers=True))
    services.credentials = CredentialStore(tmp_path / "credentials.json")
    services.credentials.set("anthropic", ANTHROPIC_KEY)

    assert services.refresh_llm() == "mock"
