"""Credential storage, masking, and precedence.

The security-relevant properties are pinned as tests rather than left to
review: a stored key must never come back out, the environment must win over
the store, and a corrupt file must not stop the app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from koe.providers.credentials import (
    KNOWN_PROVIDERS,
    CredentialError,
    CredentialStore,
    Source,
    Status,
    fingerprint,
)

ANTHROPIC_KEY = "sk-ant-api03-" + "T3stK3y" * 8
OPENAI_KEY = "sk-proj-" + "T3stK3y" * 8


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CredentialStore:
    for spec in KNOWN_PROVIDERS:
        monkeypatch.delenv(spec.env_var, raising=False)
    return CredentialStore(tmp_path / "credentials.json")


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def test_a_fingerprint_never_contains_the_key() -> None:
    """Read-back of a written secret is how a UI bug becomes a leak."""
    mark = fingerprint(ANTHROPIC_KEY)
    assert ANTHROPIC_KEY not in mark
    assert len(mark) < 20
    assert mark.startswith("sk-ant-")
    assert mark.endswith(ANTHROPIC_KEY[-4:])


def test_a_short_key_is_still_masked() -> None:
    assert "abc" not in fingerprint("abcdef")


def test_an_empty_key_has_no_fingerprint() -> None:
    assert fingerprint("") == ""


def test_the_description_carries_no_secret(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    payload = json.dumps(store.describe("anthropic").to_dict())
    assert ANTHROPIC_KEY not in payload
    assert "sk-ant-api03" not in payload.replace(fingerprint(ANTHROPIC_KEY), "")


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def test_round_trip(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    assert store.resolve("anthropic") == ANTHROPIC_KEY


def test_nothing_configured(store: CredentialStore) -> None:
    info = store.describe("anthropic")
    assert not info.configured
    assert info.source is Source.NONE
    assert info.fingerprint == ""


def test_a_key_is_not_stored_in_plaintext(store: CredentialStore) -> None:
    """Obfuscated at minimum; encrypted where the platform provides it."""
    store.set("anthropic", ANTHROPIC_KEY)
    raw = store.path.read_text(encoding="utf-8")
    assert ANTHROPIC_KEY not in raw


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_windows_uses_dpapi(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    stored = json.loads(store.path.read_text(encoding="utf-8"))
    assert stored["anthropic"]["scheme"] == "dpapi"
    assert store.resolve("anthropic") == ANTHROPIC_KEY


def test_delete(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    info = store.delete("anthropic")
    assert not info.configured
    assert store.resolve("anthropic") is None


def test_deleting_something_absent_is_fine(store: CredentialStore) -> None:
    assert not store.delete("anthropic").configured


def test_a_corrupt_file_is_treated_as_empty(store: CredentialStore) -> None:
    """A user can re-enter a key; they cannot recover from a dead app."""
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"anthropic": {"scheme":', encoding="utf-8")
    assert CredentialStore(store.path).resolve("anthropic") is None


def test_two_providers_are_independent(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    store.set("openai", OPENAI_KEY)
    assert store.resolve("anthropic") == ANTHROPIC_KEY
    assert store.resolve("openai") == OPENAI_KEY


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_an_empty_key_is_rejected(store: CredentialStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.set("anthropic", "   ")


def test_a_wrong_shaped_key_is_rejected_immediately(store: CredentialStore) -> None:
    """Catching an obviously wrong paste beats a confusing 401 later."""
    with pytest.raises(CredentialError) as caught:
        store.set("anthropic", "definitely-not-a-key-but-long-enough-to-pass-length")
    assert caught.value.code == "bad_prefix"
    assert caught.value.context["prefix"] == "sk-ant-"


def test_an_unknown_provider_is_rejected(store: CredentialStore) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        store.set("nonesuch", ANTHROPIC_KEY)


def test_surrounding_whitespace_is_forgiven(store: CredentialStore) -> None:
    """People paste with a trailing newline; that is not a user error."""
    store.set("anthropic", f"  {ANTHROPIC_KEY}\n")
    assert store.resolve("anthropic") == ANTHROPIC_KEY


# --------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------


def test_the_environment_wins_over_the_store(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment injecting secrets must not be overridden by a stale file."""
    store.set("anthropic", ANTHROPIC_KEY)
    monkeypatch.setenv("KOE_ANTHROPIC_API_KEY", "sk-ant-from-the-environment-xxxx")

    assert store.resolve("anthropic") == "sk-ant-from-the-environment-xxxx"
    assert store.source_of("anthropic") is Source.ENVIRONMENT


def test_an_environment_credential_is_not_editable(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KOE_ANTHROPIC_API_KEY", "sk-ant-from-the-environment-xxxx")
    info = store.describe("anthropic")
    assert info.source is Source.ENVIRONMENT
    assert not info.editable
    assert info.configured


# --------------------------------------------------------------------------
# verification state
# --------------------------------------------------------------------------


def test_a_new_key_is_unverified(store: CredentialStore) -> None:
    """Claiming 'valid' on save would assert something nothing checked."""
    assert store.set("anthropic", ANTHROPIC_KEY).status is Status.UNKNOWN


def test_a_verification_result_persists(store: CredentialStore) -> None:
    store.set("anthropic", ANTHROPIC_KEY)
    store.record_verification("anthropic", Status.VALID, "accepted")

    reloaded = CredentialStore(store.path).describe("anthropic")
    assert reloaded.status is Status.VALID
    assert reloaded.verified_at is not None
    assert reloaded.detail == "accepted"


def test_describe_all_lists_every_known_provider(store: CredentialStore) -> None:
    listed = {info.provider for info in store.describe_all()}
    assert listed == {spec.id for spec in KNOWN_PROVIDERS}
