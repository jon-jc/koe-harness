"""The updater: what it will install, and what it refuses to.

Most of this file is refusals, because that is where an updater's risk lives.
A bug that fails to install an update costs someone a newer version; a bug
that installs the wrong thing costs them their machine.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from koe import __version__
from koe.desktop import paths, update_api
from koe.desktop.settings import DesktopSettings
from koe.desktop.update_api import build_update_router
from koe.desktop.updater import (
    BuildInfo,
    UpdateError,
    Updater,
    is_newer,
    load_build_info,
    parse_manifest,
    parse_version,
)

REPO = "jon-jc/koe-harness"
PREFIX = f"https://github.com/{REPO}/releases/download/"
PAYLOAD = b"MZ" + b"koe installer bytes " * 2000


@pytest.fixture(autouse=True)
def _no_feed_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KOE_UPDATE_FEED", raising=False)


def manifest(
    version: str = "0.1.30", *, payload: bytes = PAYLOAD, **installer: Any
) -> dict[str, Any]:
    name = f"koe-setup-{version}.exe"
    body: dict[str, Any] = {
        "name": name,
        "url": f"{PREFIX}v{version}/{name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    body.update(installer)
    return {"version": version, "tag": f"v{version}", "installer": body}


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeGitHub:
    """A release, its manifest and its installer, served from memory."""

    def __init__(
        self,
        release: dict[str, Any] | None = None,
        *,
        payload: bytes = PAYLOAD,
        status: int | None = None,
        tag: str | None = None,
    ) -> None:
        self.release = release if release is not None else manifest()
        self.payload = payload
        self.status = status
        self.tag = tag
        self.requests: list[str] = []

    def __call__(self, request: Any, timeout: float = 0) -> Response:
        url = request.full_url
        self.requests.append(url)
        if self.status is not None:
            raise urllib.error.HTTPError(url, self.status, "refused", {}, None)  # type: ignore[arg-type]
        if url.endswith("/releases/latest"):
            tag = self.tag or self.release["tag"]
            asset = {"name": "latest.json", "browser_download_url": f"{PREFIX}{tag}/latest.json"}
            return Response(json.dumps({"tag_name": tag, "assets": [asset]}).encode())
        if url.endswith("/latest.json"):
            return Response(json.dumps(self.release).encode())
        if url.endswith(".exe"):
            return Response(self.payload)
        raise AssertionError(f"unexpected request: {url}")

    @property
    def downloaded(self) -> bool:
        return any(url.endswith(".exe") for url in self.requests)


class Launcher:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> None:
        self.commands.append(command)


def make_updater(
    tmp_path: Path,
    opener: Any,
    *,
    version: str = "0.1.29",
    launcher: Launcher | None = None,
    supported: bool = True,
    auto: bool = True,
) -> Updater:
    return Updater(
        info=BuildInfo(version=version, channel="release"),
        directory=tmp_path / "updates",
        supported=supported,
        auto=auto,
        opener=opener,
        launcher=launcher or Launcher(),
    )


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------


def test_versions_compare_as_numbers_not_strings() -> None:
    assert is_newer("0.1.10", "0.1.9")
    assert is_newer("v0.2.0", "0.1.99")
    assert not is_newer("0.1.29", "0.1.29")
    assert not is_newer("0.1.28", "0.1.29")


def test_an_unparseable_version_is_never_newer() -> None:
    """Guessing how to order a suffix is how an updater installs an older build."""
    assert parse_version("0.1.30-beta") is None
    assert not is_newer("0.1.30-beta", "0.1.29")
    assert not is_newer("0.1.30", "nightly")


def test_the_build_stamp_is_read_from_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = {"version": "0.1.31", "commit": "abc123", "channel": "release"}
    (tmp_path / "build_info.json").write_text(json.dumps(stamp), encoding="utf-8")
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path)
    info = load_build_info()
    assert (info.version, info.commit, info.channel) == ("0.1.31", "abc123", "release")


def test_a_malformed_stamp_falls_back_to_the_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "build_info.json").write_text(json.dumps({"version": "latest"}), encoding="utf-8")
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path)
    assert load_build_info().version == __version__


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


def test_a_well_formed_manifest_is_accepted() -> None:
    release = parse_manifest(manifest(), trusted_prefix=PREFIX)
    assert release.version == "0.1.30"
    assert release.url == f"{PREFIX}v0.1.30/koe-setup-0.1.30.exe"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"url": "https://evil.example/koe-setup-0.1.30.exe"}, "hosted"),
        ({"url": f"{PREFIX}v0.1.30/../../../../someone/else/koe-setup-0.1.30.exe"}, "hosted"),
        ({"url": f"{PREFIX}v0.1.30/koe-setup-0.1.30.exe?redirect=elsewhere"}, "hosted"),
        ({"url": f"{PREFIX}v0.1.30/%2e%2e/koe-setup-0.1.30.exe"}, "hosted"),
        ({"url": f"{PREFIX}v0.1.30/something-else.exe"}, "hosted"),
        ({"url": f"{PREFIX}latest/koe-setup-0.1.30.exe"}, "hosted"),
        ({"name": "setup.exe", "url": f"{PREFIX}v0.1.30/setup.exe"}, "name"),
        ({"sha256": "not-a-digest"}, "SHA-256"),
        ({"size": 0}, "size"),
        ({"size": True}, "size"),
        ({"size": 10 * 1024 * 1024 * 1024}, "size"),
    ],
)
def test_a_suspicious_manifest_is_refused(change: dict[str, Any], reason: str) -> None:
    with pytest.raises(UpdateError, match=reason):
        parse_manifest(manifest(**change), trusted_prefix=PREFIX)


def test_a_manifest_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(UpdateError):
        parse_manifest(["not", "an", "object"], trusted_prefix=PREFIX)


# --------------------------------------------------------------------------
# checking and downloading
# --------------------------------------------------------------------------


def test_a_newer_release_is_downloaded_and_verified(tmp_path: Path) -> None:
    status = make_updater(tmp_path, FakeGitHub()).check()
    assert status.state == "ready"
    assert status.available == "0.1.30"
    assert status.progress == 1.0
    assert (tmp_path / "updates" / "koe-setup-0.1.30.exe").read_bytes() == PAYLOAD


def test_the_same_version_is_current_and_nothing_is_downloaded(tmp_path: Path) -> None:
    github = FakeGitHub(manifest("0.1.29"))
    assert make_updater(tmp_path, github).check().state == "current"
    assert not github.downloaded


def test_an_older_release_is_never_installed(tmp_path: Path) -> None:
    github = FakeGitHub(manifest("0.1.20"))
    assert make_updater(tmp_path, github).check().state == "current"
    assert not github.downloaded


def test_a_download_that_does_not_match_its_digest_is_discarded(tmp_path: Path) -> None:
    """Bytes that are not the published bytes never reach disk under the installer's name."""
    github = FakeGitHub(payload=PAYLOAD[:-1] + b"X")
    status = make_updater(tmp_path, github).check()
    assert status.state == "error"
    assert "SHA-256" in status.error
    assert list((tmp_path / "updates").iterdir()) == []


def test_a_download_longer_than_the_manifest_says_is_stopped(tmp_path: Path) -> None:
    status = make_updater(tmp_path, FakeGitHub(payload=PAYLOAD + b"extra")).check()
    assert status.state == "error"
    assert "larger" in status.error
    assert list((tmp_path / "updates").iterdir()) == []


def test_a_truncated_download_is_refused(tmp_path: Path) -> None:
    status = make_updater(tmp_path, FakeGitHub(payload=PAYLOAD[:100])).check()
    assert status.state == "error"
    assert "ended early" in status.error


def test_no_published_release_is_not_an_error(tmp_path: Path) -> None:
    assert make_updater(tmp_path, FakeGitHub(status=404)).check().state == "current"


def test_the_rate_limit_is_reported_and_left_for_the_next_check(tmp_path: Path) -> None:
    status = make_updater(tmp_path, FakeGitHub(status=403)).check()
    assert status.state == "error"
    assert "rate limit" in status.error


def test_a_network_failure_is_a_status_not_a_crash(tmp_path: Path) -> None:
    def offline(request: Any, timeout: float = 0) -> Response:
        raise urllib.error.URLError("no route to host")

    status = make_updater(tmp_path, offline).check()
    assert status.state == "error"
    assert "reach" in status.error


def test_a_manifest_describing_a_different_release_is_refused(tmp_path: Path) -> None:
    status = make_updater(tmp_path, FakeGitHub(tag="v0.1.31")).check()
    assert status.state == "error"
    assert "v0.1.31" in status.error


def test_a_source_checkout_never_updates(tmp_path: Path) -> None:
    github = FakeGitHub()
    updater = make_updater(tmp_path, github, supported=False)
    assert updater.check().state == "disabled"
    assert github.requests == []


def test_a_cleartext_remote_feed_is_ignored(tmp_path: Path) -> None:
    """Plain HTTP from a remote host could be answered by anyone on the path."""
    updater = Updater(
        info=BuildInfo(version="0.1.29"),
        directory=tmp_path,
        supported=True,
        feed="http://updates.example.com/latest.json",
    )
    assert updater.feed is None


def test_a_loopback_feed_serves_its_own_installer(tmp_path: Path) -> None:
    feed = "http://127.0.0.1:8900/latest.json"
    local = manifest(url="http://127.0.0.1:8900/koe-setup-0.1.30.exe")

    def serve(request: Any, timeout: float = 0) -> Response:
        if request.full_url == feed:
            return Response(json.dumps(local).encode())
        if request.full_url.endswith(".exe"):
            return Response(PAYLOAD)
        raise AssertionError(request.full_url)

    updater = Updater(
        info=BuildInfo(version="0.1.29"),
        directory=tmp_path / "updates",
        supported=True,
        feed=feed,
        opener=serve,
        launcher=Launcher(),
    )
    assert updater.check().state == "ready"


def test_installed_and_superseded_installers_are_pruned(tmp_path: Path) -> None:
    directory = tmp_path / "updates"
    directory.mkdir()
    for name in (
        "koe-setup-0.1.28.exe",
        "koe-setup-0.1.29.exe",
        "koe-setup-0.1.30.exe",
        "notes.txt",
    ):
        (directory / name).write_bytes(b"x")
    make_updater(tmp_path, FakeGitHub()).prune()
    assert sorted(path.name for path in directory.iterdir()) == [
        "koe-setup-0.1.30.exe",
        "notes.txt",
    ]


# --------------------------------------------------------------------------
# installing
# --------------------------------------------------------------------------


def test_apply_hands_over_to_a_silent_installer_waiting_for_this_process(tmp_path: Path) -> None:
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)
    updater.check()
    updater.apply(relaunch=True)

    [command] = launcher.commands
    assert command[0].endswith("koe-setup-0.1.30.exe")
    assert "/VERYSILENT" in command
    assert f"/waitpid={os.getpid()}" in command
    assert "/relaunch=1" in command
    assert updater.snapshot().state == "applying"


def test_nothing_is_launched_before_an_update_is_ready(tmp_path: Path) -> None:
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(manifest("0.1.29")), launcher=launcher)
    updater.check()
    with pytest.raises(UpdateError, match="no update"):
        updater.apply(relaunch=True)
    assert launcher.commands == []


def test_an_installer_changed_after_verification_is_not_run(tmp_path: Path) -> None:
    """It sat in a user-writable directory between the check and the click."""
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)
    updater.check()
    installer = tmp_path / "updates" / "koe-setup-0.1.30.exe"
    installer.write_bytes(b"MZ not what was verified")

    with pytest.raises(UpdateError, match="changed on disk"):
        updater.apply(relaunch=False)
    assert launcher.commands == []
    assert not installer.exists()
    assert updater.snapshot().state == "error"


def test_a_ready_update_installs_on_exit_only_when_automatic(tmp_path: Path) -> None:
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher, auto=False)
    updater.check()

    assert updater.apply_on_exit() is False
    assert launcher.commands == []

    updater.set_auto(True)
    assert updater.apply_on_exit() is True
    assert "/relaunch=0" in launcher.commands[0]


# --------------------------------------------------------------------------
# the HTTP surface
# --------------------------------------------------------------------------


class ImmediateTimer:
    """Runs the exit callback at once instead of after a delay."""

    def __init__(self, delay: float, function: Any) -> None:
        self.function = function

    def start(self) -> None:
        self.function()


def client_for(
    updater: Updater, exits: list[bool], remembered: list[bool] | None = None
) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_update_router(
            updater,
            on_apply=lambda: exits.append(True),
            on_auto_changed=(remembered.append if remembered is not None else lambda enabled: None),
        )
    )
    return TestClient(app)


def test_the_status_is_readable(tmp_path: Path) -> None:
    body = client_for(make_updater(tmp_path, FakeGitHub()), []).get("/v1/update").json()
    assert body["state"] == "idle"
    assert body["current"] == "0.1.29"


def test_a_check_through_the_api_downloads(tmp_path: Path) -> None:
    client = client_for(make_updater(tmp_path, FakeGitHub()), [])
    assert client.post("/v1/update/check").json()["state"] == "ready"


def test_a_cross_origin_request_cannot_start_an_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any page open in the person's browser can send requests to loopback."""
    monkeypatch.setattr(update_api.threading, "Timer", ImmediateTimer)
    exits: list[bool] = []
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)
    updater.check()

    response = client_for(updater, exits).post(
        "/v1/update/apply", json={"relaunch": True}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403
    assert launcher.commands == []
    assert exits == []


def test_apply_through_the_api_launches_and_then_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_api.threading, "Timer", ImmediateTimer)
    exits: list[bool] = []
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)
    updater.check()

    response = client_for(updater, exits).post(
        "/v1/update/apply", json={"relaunch": True}, headers={"Origin": "http://testserver"}
    )
    assert response.status_code == 200
    assert len(launcher.commands) == 1
    assert exits == [True]


def test_apply_without_a_ready_update_is_a_conflict(tmp_path: Path) -> None:
    client = client_for(make_updater(tmp_path, FakeGitHub()), [])
    assert client.post("/v1/update/apply", json={}).status_code == 409


def test_turning_off_automatic_updates_is_remembered(tmp_path: Path) -> None:
    remembered: list[bool] = []
    client = client_for(make_updater(tmp_path, FakeGitHub()), [], remembered)
    body = client.post("/v1/update/auto", json={"enabled": False}).json()
    assert body["auto"] is False
    assert remembered == [False]


def test_automatic_updates_default_on_and_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "settings.json"
    monkeypatch.setattr(DesktopSettings, "path", classmethod(lambda cls: target))
    settings = DesktopSettings.load()
    assert settings.auto_update is True
    settings.auto_update = False
    settings.save()
    assert DesktopSettings.load().auto_update is False


def test_the_installer_may_close_anything_still_holding_its_files(tmp_path: Path) -> None:
    """An orphaned console host from a crashed older build would otherwise block the install."""
    launcher = Launcher()
    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)
    updater.check()
    updater.apply(relaunch=False)
    assert "/FORCECLOSEAPPLICATIONS" in launcher.commands[0]


@pytest.mark.skipif(os.name != "nt", reason="Windows process-creation flags")
def test_the_installer_is_started_outside_the_apps_job(tmp_path: Path) -> None:
    import subprocess

    calls: list[dict[str, Any]] = []

    def launcher(command: list[str], **kwargs: Any) -> None:
        calls.append(kwargs)

    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)  # type: ignore[arg-type]
    updater.check()
    updater.apply(relaunch=True)
    assert calls[0]["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB


@pytest.mark.skipif(os.name != "nt", reason="Windows process-creation flags")
def test_a_job_that_forbids_breakaway_still_gets_its_installer(tmp_path: Path) -> None:
    import subprocess

    calls: list[dict[str, Any]] = []

    def launcher(command: list[str], **kwargs: Any) -> None:
        calls.append(kwargs)
        if kwargs["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB:
            raise PermissionError("access is denied")

    updater = make_updater(tmp_path, FakeGitHub(), launcher=launcher)  # type: ignore[arg-type]
    updater.check()
    updater.apply(relaunch=True)
    assert len(calls) == 2
    assert not calls[1]["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB
    assert updater.snapshot().state == "applying"
