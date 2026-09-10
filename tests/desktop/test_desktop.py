"""Desktop application internals.

The window itself is not testable here, but everything underneath it is — and
the things underneath are where desktop-specific bugs live: port binding,
stale locks, settings files that were truncated by a power cut, and paths that
only move once the app is frozen.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from koe.desktop import paths
from koe.desktop.instance import AlreadyRunning, InstanceLock, _process_alive
from koe.desktop.server import EmbeddedServer, reserve_loopback_port
from koe.desktop.settings import DesktopSettings, WindowState

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


@pytest.mark.real_user_paths
def test_writable_dirs_are_not_next_to_the_executable() -> None:
    """A packaged app may live in Program Files, which is read-only.

    Opts out of the suite-wide config sandbox: this test is about where
    the real directories are, so a redirected one proves nothing.
    """
    for directory in (paths.config_dir(), paths.data_dir(), paths.log_dir()):
        assert directory.is_absolute()
        assert paths.APP_NAME in directory.parts


def test_resource_root_points_at_the_web_client() -> None:
    assert (paths.web_root() / "index.html").exists()


def test_ensure_dirs_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    paths.ensure_dirs()
    paths.ensure_dirs()


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


@pytest.fixture
def settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "settings.json"
    monkeypatch.setattr(DesktopSettings, "path", classmethod(lambda cls: target))
    return target


def test_defaults_when_absent(settings_path: Path) -> None:
    settings = DesktopSettings.load()
    assert settings.window.width >= 720
    assert settings.ui_language == "ja"


def test_round_trip(settings_path: Path) -> None:
    original = DesktopSettings(
        window=WindowState(width=1400, height=900, x=100, y=50),
        ui_language="en",
        theme="dark",
    )
    original.save()

    loaded = DesktopSettings.load()

    assert loaded.window.width == 1400
    assert loaded.ui_language == "en"
    assert loaded.theme == "dark"


def test_truncated_json_falls_back_to_defaults(settings_path: Path) -> None:
    """A power cut mid-write must not stop the app from opening."""
    settings_path.write_text('{"window": {"width": 12', encoding="utf-8")
    assert DesktopSettings.load().ui_language == "ja"


def test_a_non_object_file_falls_back(settings_path: Path) -> None:
    settings_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert DesktopSettings.load().theme == "system"


def test_unknown_keys_are_ignored_not_rejected(settings_path: Path) -> None:
    """A file written by a newer version must still open in an older one."""
    settings_path.write_text(
        json.dumps({"ui_language": "en", "future_feature": {"nested": True}}),
        encoding="utf-8",
    )
    assert DesktopSettings.load().ui_language == "en"


def test_wrong_types_are_ignored(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({"ui_language": 42}), encoding="utf-8")
    assert DesktopSettings.load().ui_language == "ja"


def test_an_off_screen_window_is_recentred() -> None:
    """Restoring onto an unplugged monitor leaves the app running but invisible."""
    restored = WindowState(width=1200, height=800, x=9_999_999, y=9_999_999).sanitized()
    assert restored.x is None
    assert restored.y is None


def test_a_tiny_window_is_clamped() -> None:
    assert WindowState(width=10, height=10).sanitized().width >= 720


def test_saving_is_atomic(settings_path: Path) -> None:
    """No half-written file is left behind for the next launch to choke on."""
    DesktopSettings(ui_language="en").save()
    assert settings_path.exists()
    assert not settings_path.with_suffix(".json.tmp").exists()
    json.loads(settings_path.read_text(encoding="utf-8"))


def test_an_unwritable_location_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing a window size is not worth taking down a live session."""
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        DesktopSettings, "path", classmethod(lambda cls: blocked / "nested" / "settings.json")
    )
    DesktopSettings().save()  # must not raise


# --------------------------------------------------------------------------
# single instance
# --------------------------------------------------------------------------


def test_a_lock_is_held_and_released(tmp_path: Path) -> None:
    lock = InstanceLock(tmp_path / "koe.lock")
    with lock:
        assert lock.path.exists()
        assert lock.path.read_text(encoding="utf-8").strip() == str(os.getpid())
    assert not lock.path.exists()


def test_a_second_instance_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "koe.lock"
    with InstanceLock(path):
        with pytest.raises(AlreadyRunning) as excinfo:
            InstanceLock(path).acquire()
        assert excinfo.value.pid == os.getpid()


def test_a_stale_lock_is_taken_over(tmp_path: Path) -> None:
    """Otherwise a crash means the app can never start again."""
    path = tmp_path / "koe.lock"
    # A PID that cannot be running: max PID + 1 on any platform of interest.
    path.write_text("4294967290", encoding="utf-8")

    lock = InstanceLock(path)
    lock.acquire()

    assert path.read_text(encoding="utf-8").strip() == str(os.getpid())
    lock.release()


def test_a_corrupt_lock_is_taken_over(tmp_path: Path) -> None:
    path = tmp_path / "koe.lock"
    path.write_text("not-a-pid", encoding="utf-8")
    lock = InstanceLock(path)
    lock.acquire()
    lock.release()


def test_release_does_not_remove_someone_elses_lock(tmp_path: Path) -> None:
    path = tmp_path / "koe.lock"
    lock = InstanceLock(path)
    lock.acquire()
    # Simulate another process winning a takeover race.
    path.write_text("4294967290", encoding="utf-8")
    lock.release()
    assert path.exists()


def test_process_liveness() -> None:
    assert _process_alive(os.getpid())
    assert not _process_alive(0)
    assert not _process_alive(-1)


def test_an_out_of_range_pid_is_not_alive() -> None:
    """A lock file is untrusted input.

    On Linux os.kill raises OverflowError rather than returning false for a
    value beyond pid_t, so a corrupt lock would crash the app at startup
    instead of being treated as stale.
    """
    assert not _process_alive(4_294_967_290)
    assert not _process_alive(2**63)


# --------------------------------------------------------------------------
# embedded server
# --------------------------------------------------------------------------


def test_a_port_is_reserved_before_the_server_starts() -> None:
    """Knowing the URL with certainty is what avoids a startup race."""
    sock, port = reserve_loopback_port()
    try:
        assert 1024 < port < 65536
        assert sock.getsockname() == ("127.0.0.1", port)
    finally:
        sock.close()


def test_reserved_ports_do_not_collide() -> None:
    a_sock, a_port = reserve_loopback_port()
    b_sock, b_port = reserve_loopback_port()
    try:
        assert a_port != b_port
    finally:
        a_sock.close()
        b_sock.close()


def test_the_reservation_actually_holds_the_port() -> None:
    """A reserve-then-close design would leave a window for a collision."""
    sock, port = reserve_loopback_port()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other, pytest.raises(OSError):
            other.bind(("127.0.0.1", port))
    finally:
        sock.close()


def test_the_server_starts_serves_and_stops() -> None:
    """The desktop runs the same app the deployment does, not a stub."""
    import urllib.request

    from koe.api.app import Services, create_app
    from koe.config import Settings

    server = EmbeddedServer(create_app(Services.default(Settings(environment="local"))))
    server.start()
    try:
        assert server.running
        with urllib.request.urlopen(f"{server.url}/health", timeout=10) as response:
            body = json.loads(response.read())
        assert body["status"] == "ok"
    finally:
        server.stop()

    assert not server.running


def test_stopping_a_server_that_never_started_is_safe() -> None:
    from koe.api.app import Services, create_app
    from koe.config import Settings

    server = EmbeddedServer(create_app(Services.default(Settings(environment="local"))))
    server.stop()  # must not raise
