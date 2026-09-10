"""Keeping the installed app current.

Every merge to ``main`` publishes a release (``.github/workflows/release.yml``):
a versioned installer, its SHA-256, and ``latest.json`` describing both. This
module is the other half — it notices the release, fetches the installer, and
hands over to it.

Five decisions carry most of the weight:

**Nothing runs that was not verified.** The manifest names the installer's
SHA-256 and exact size. The download is hashed while it streams, a file that
does not match is deleted before anything can execute it, and the hash is
checked *again* immediately before launch, because the file has been sitting in
a user-writable directory since.

**Only this repository's releases.** An installer URL has to be
``https://github.com/<repo>/releases/download/<tag>/<installer>``. A manifest
pointing anywhere else is refused, so a tampered manifest cannot redirect the
download even though it can name a digest.

**Downloading is background work; installing waits for the person.**
"Seamless" means nobody waits on a download or runs an installer by hand — not
that the app restarts under someone mid-meeting. A ready update installs when
the app is closed, or at once from "Restart to update".

**The installer replaces files only after koe has gone.** It is started with
this process's PID and waits for it to exit (see ``packaging/koe.iss``), so
there is no moment where the installer and the running app both hold the same
files.

**Failure is quiet and recorded.** No network, GitHub's rate limit, a bad
digest: each is logged, shown as a status in Settings → About, and retried at
the next check. An updater that nags or crashes is worse than none.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from koe import __version__
from koe.desktop import paths

logger = logging.getLogger(__name__)

REPOSITORY = "jon-jc/koe-harness"
MANIFEST_NAME = "latest.json"
BUILD_INFO_NAME = "build_info.json"

#: Soon after launch, but not during it: startup is when the machine is busiest
#: and the person is waiting for a window, not for a version check.
FIRST_CHECK_DELAY_S = 20.0
CHECK_INTERVAL_S = 6 * 60 * 60
HTTP_TIMEOUT_S = 20.0
#: The real installer is ~50 MB. Anything past this is not a koe release.
MAX_INSTALLER_BYTES = 600 * 1024 * 1024
CHUNK_BYTES = 256 * 1024

State = Literal[
    "disabled", "idle", "checking", "current", "downloading", "ready", "applying", "error"
]

DIGEST = re.compile(r"[0-9a-f]{64}")
INSTALLER_NAME = re.compile(r"koe-setup-\d+(?:\.\d+){1,3}\.exe")
TAG = re.compile(r"v\d+(?:\.\d+){1,3}")


class UpdateError(RuntimeError):
    """An update could not proceed. The message is shown to the user."""


class NoReleases(UpdateError):
    """The repository has not published a release yet."""


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    """``v0.1.29`` → ``(0, 1, 29)``; ``None`` for anything else.

    Deliberately narrow. A pre-release suffix or a date is not something this
    project publishes, and an updater that guessed how to order one would be
    an updater that could install an older build over a newer one.
    """
    match = re.fullmatch(r"v?(\d+(?:\.\d+){0,3})", text.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is strictly newer. Unparseable is never newer."""
    new, old = parse_version(candidate), parse_version(current)
    if new is None or old is None:
        return False
    width = max(len(new), len(old))
    return new + (0,) * (width - len(new)) > old + (0,) * (width - len(old))


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """What this build is, as stamped by ``packaging/build.py``."""

    version: str
    commit: str = ""
    #: ``release`` from CI, ``local`` from a developer's build, ``source`` when
    #: running from a checkout.
    channel: str = "source"

    @property
    def installed(self) -> bool:
        """Only a frozen Windows build can be replaced by an installer."""
        return paths.is_frozen() and sys.platform == "win32"


def load_build_info() -> BuildInfo:
    """Read the stamp from the bundle, falling back to the package version."""
    try:
        raw = json.loads((paths.resource_root() / BUILD_INFO_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return BuildInfo(version=__version__)
    if not isinstance(raw, dict):
        return BuildInfo(version=__version__)
    version = str(raw.get("version") or __version__)
    if parse_version(version) is None:
        version = __version__
    return BuildInfo(
        version=version,
        commit=str(raw.get("commit") or ""),
        channel=str(raw.get("channel") or "local"),
    )


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    tag: str
    name: str
    url: str
    sha256: str
    size: int
    notes_url: str = ""


def github_download_prefix(repository: str = REPOSITORY) -> str:
    return f"https://github.com/{repository}/releases/download/"


def parse_manifest(raw: object, *, trusted_prefix: str) -> Release:
    """Validate ``latest.json``. Anything suspicious is refused, not repaired."""
    if not isinstance(raw, dict):
        raise UpdateError("the release manifest is not an object")

    version = str(raw.get("version", ""))
    if parse_version(version) is None:
        raise UpdateError(f"the release manifest's version {version!r} is not a version")

    installer = raw.get("installer")
    if not isinstance(installer, dict):
        raise UpdateError("the release manifest names no installer")

    name = str(installer.get("name", ""))
    url = str(installer.get("url", ""))
    sha256 = str(installer.get("sha256", "")).lower()
    size = installer.get("size")

    if not INSTALLER_NAME.fullmatch(name):
        raise UpdateError(f"unexpected installer name {name!r}")
    if not DIGEST.fullmatch(sha256):
        raise UpdateError("the release manifest carries no valid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_INSTALLER_BYTES:
        raise UpdateError("the release manifest carries no plausible installer size")
    if not _trusted_url(url, name=name, prefix=trusted_prefix):
        raise UpdateError("the installer is not hosted by this project's releases")

    return Release(
        version=version,
        tag=str(raw.get("tag") or f"v{version}"),
        name=name,
        url=url,
        sha256=sha256,
        size=size,
        notes_url=str(raw.get("notes_url") or ""),
    )


def _trusted_url(url: str, *, name: str, prefix: str) -> bool:
    """Whether ``url`` is exactly ``<prefix><tag>/<name>`` with nothing smuggled in.

    A prefix match alone would accept ``…/download/../../../elsewhere``, and a
    query string can change what a server returns without changing the path.
    """
    if not url.startswith(prefix) or any(mark in url for mark in ("?", "#", "\\", "%")):
        return False
    rest = url[len(prefix) :]
    segments = rest.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return False
    if prefix.startswith("https://github.com/"):
        return len(segments) == 2 and bool(TAG.fullmatch(segments[0])) and segments[1] == name
    return segments[-1] == name


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


@dataclass(slots=True)
class UpdateStatus:
    state: State
    current: str
    channel: str
    auto: bool
    available: str = ""
    progress: float = 0.0
    error: str = ""
    checked_at: float = 0.0
    notes_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Opener = Callable[..., Any]
Launcher = Callable[..., Any]


# --------------------------------------------------------------------------
# the updater
# --------------------------------------------------------------------------


class Updater:
    """Checks for, downloads, verifies, and hands over to a newer installer."""

    def __init__(
        self,
        *,
        info: BuildInfo | None = None,
        directory: Path | None = None,
        auto: bool = True,
        feed: str | None = None,
        repository: str = REPOSITORY,
        supported: bool | None = None,
        opener: Opener = urllib.request.urlopen,
        launcher: Launcher = subprocess.Popen,
    ) -> None:
        self.info = info or load_build_info()
        self.directory = directory or (paths.data_dir() / "updates")
        self.repository = repository
        # An explicit feed is for testing a release before it is published. It
        # must be HTTPS, or plain HTTP on loopback — never a remote cleartext
        # server, which anyone on the network path could answer.
        self.feed = (feed if feed is not None else os.environ.get("KOE_UPDATE_FEED", "")) or None
        if self.feed is not None and not _acceptable_feed(self.feed):
            logger.warning("ignoring KOE_UPDATE_FEED %r: not https or loopback", self.feed)
            self.feed = None
        self.supported = self.info.installed if supported is None else supported

        self._opener = opener
        self._launcher = launcher
        self._lock = threading.RLock()
        self._busy = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._release: Release | None = None
        self._installer: Path | None = None
        self._status = UpdateStatus(
            state="idle" if self.supported else "disabled",
            current=self.info.version,
            channel=self.info.channel,
            auto=auto,
        )

    # -- status --------------------------------------------------------------

    def snapshot(self) -> UpdateStatus:
        with self._lock:
            return UpdateStatus(**self._status.to_dict())

    def _set(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)

    @property
    def auto(self) -> bool:
        return self.snapshot().auto

    def set_auto(self, enabled: bool) -> None:
        self._set(auto=bool(enabled))

    # -- background ----------------------------------------------------------

    def start(self) -> None:
        """Check soon after launch and periodically after that, if enabled."""
        if not self.supported or self._thread is not None:
            return
        self.prune()
        self._thread = threading.Thread(target=self._loop, name="koe.updater", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        if self._stop.wait(FIRST_CHECK_DELAY_S):
            return
        while not self._stop.is_set():
            if self.auto:
                self.check()
            if self._stop.wait(CHECK_INTERVAL_S):
                return

    # -- check and download --------------------------------------------------

    def check(self) -> UpdateStatus:
        """Look for a newer release and, if there is one, download it.

        Synchronous; the background loop and the API both call it on a worker
        thread. A check already in progress is not repeated — the caller gets
        the status of the one that is running.
        """
        if not self.supported:
            return self.snapshot()
        if not self._busy.acquire(blocking=False):
            return self.snapshot()
        try:
            if self.snapshot().state in ("ready", "applying"):
                return self.snapshot()
            self._set(state="checking", error="")
            try:
                release = self._latest()
            except NoReleases:
                self._set(state="current", available="", checked_at=time.time())
                return self.snapshot()
            except (UpdateError, OSError, ValueError) as exc:
                self._fail(_describe(exc))
                return self.snapshot()

            if not is_newer(release.version, self.info.version):
                self._set(state="current", available="", checked_at=time.time())
                return self.snapshot()

            logger.info("update available", extra={"version": release.version})
            self._set(
                state="downloading",
                available=release.version,
                progress=0.0,
                notes_url=release.notes_url,
            )
            try:
                installer = self._download(release)
            except (UpdateError, OSError, ValueError) as exc:
                self._fail(_describe(exc))
                return self.snapshot()

            with self._lock:
                self._release = release
                self._installer = installer
            self._set(state="ready", progress=1.0, checked_at=time.time())
            logger.info(
                "update ready", extra={"version": release.version, "installer": str(installer)}
            )
            return self.snapshot()
        finally:
            self._busy.release()

    def _fail(self, message: str) -> None:
        logger.warning("update check failed: %s", message)
        self._set(state="error", error=message, progress=0.0, checked_at=time.time())

    def _latest(self) -> Release:
        if self.feed is not None:
            origin = _origin(self.feed)
            return parse_manifest(self._get_json(self.feed), trusted_prefix=f"{origin}/")

        api = f"https://api.github.com/repos/{self.repository}/releases/latest"
        try:
            release = self._get_json(api, accept="application/vnd.github+json")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NoReleases("no release has been published yet") from exc
            if exc.code in (403, 429):
                raise UpdateError("GitHub's rate limit was reached; will try again later") from exc
            raise
        if not isinstance(release, dict):
            raise UpdateError("GitHub returned an unexpected response")

        raw_assets = release.get("assets")
        assets = raw_assets if isinstance(raw_assets, list) else []
        manifest_url = next(
            (
                str(asset.get("browser_download_url", ""))
                for asset in assets
                if isinstance(asset, dict) and asset.get("name") == MANIFEST_NAME
            ),
            "",
        )
        if not manifest_url.startswith(github_download_prefix(self.repository)):
            raise UpdateError("the latest release carries no update manifest")

        parsed = parse_manifest(
            self._get_json(manifest_url), trusted_prefix=github_download_prefix(self.repository)
        )
        tag = str(release.get("tag_name", ""))
        if tag and tag != parsed.tag:
            raise UpdateError(f"the manifest describes {parsed.tag}, but the release is {tag}")
        return parsed

    def _get_json(self, url: str, *, accept: str = "application/json") -> object:
        request = urllib.request.Request(
            url,
            headers={"Accept": accept, "User-Agent": f"koe-desktop/{self.info.version}"},
        )
        with self._opener(request, timeout=HTTP_TIMEOUT_S) as response:
            body = response.read(1_000_000)
        return json.loads(body)

    def _download(self, release: Release) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.directory / release.name

        # Already fetched by an earlier run that did not get to install it.
        if (
            final.exists()
            and final.stat().st_size == release.size
            and sha256_of(final) == release.sha256
        ):
            return final

        partial = self.directory / f"{release.name}.part"
        digest = hashlib.sha256()
        received = 0
        request = urllib.request.Request(
            release.url, headers={"User-Agent": f"koe-desktop/{self.info.version}"}
        )
        try:
            with (
                self._opener(request, timeout=HTTP_TIMEOUT_S) as response,
                partial.open("wb") as out,
            ):
                while True:
                    if self._stop.is_set():
                        raise UpdateError("the download was cancelled")
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > release.size:
                        raise UpdateError("the download is larger than the release says it is")
                    digest.update(chunk)
                    out.write(chunk)
                    self._set(progress=round(received / release.size, 3))
            if received != release.size:
                raise UpdateError(
                    f"the download ended early ({received:,} of {release.size:,} bytes)"
                )
            if digest.hexdigest() != release.sha256:
                raise UpdateError(
                    "the download does not match its published SHA-256 and was discarded"
                )
            partial.replace(final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        self.prune(keep=final)
        return final

    def prune(self, keep: Path | None = None) -> None:
        """Delete installers that are no longer useful.

        Anything at or below the running version has already been installed,
        and anything other than the kept file is superseded. They are ~50 MB
        each, and a directory of them is how an updater quietly eats a disk.
        """
        if not self.directory.is_dir():
            return
        for path in self.directory.iterdir():
            if keep is not None and path == keep:
                continue
            match = re.fullmatch(r"koe-setup-(\d+(?:\.\d+){1,3})\.exe(?:\.part)?", path.name)
            if match is None:
                continue
            stale = keep is not None or not is_newer(match.group(1), self.info.version)
            if stale:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.debug("could not remove %s: %s", path, exc)

    # -- install -------------------------------------------------------------

    def installer_command(self, installer: Path, *, pid: int, relaunch: bool) -> list[str]:
        """The command that installs silently and, optionally, starts koe again."""
        return [
            str(installer),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            # Anything still holding a file it must replace — a console host
            # orphaned by a crash of an older build — is closed, not waited on.
            "/FORCECLOSEAPPLICATIONS",
            f"/waitpid={pid}",
            f"/relaunch={1 if relaunch else 0}",
            f"/LOG={paths.log_dir() / 'update-install.log'}",
        ]

    def apply(self, *, relaunch: bool) -> None:
        """Start the verified installer and leave it waiting for this process.

        The caller exits afterwards; the installer does nothing until it has.
        """
        with self._lock:
            installer = self._installer
            release = self._release
            if self._status.state != "ready" or installer is None or release is None:
                raise UpdateError("no update is ready to install")

            # Checked again: the file has been sitting in a user-writable
            # directory since it was verified.
            if not installer.exists() or sha256_of(installer) != release.sha256:
                installer.unlink(missing_ok=True)
                self._installer = None
                self._fail("the downloaded installer changed on disk and was discarded")
                raise UpdateError("the downloaded installer changed on disk and was discarded")

            command = self.installer_command(installer, pid=os.getpid(), relaunch=relaunch)
            logger.info(
                "installing update", extra={"version": release.version, "relaunch": relaunch}
            )
            if sys.platform == "win32":
                # Detached, and outside koe's job object (koe.desktop.jobs). The
                # job ends koe's children when koe exits, and the installer is
                # the one child that has to outlive it.
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                try:
                    self._launcher(
                        command,
                        close_fds=True,
                        creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                    )
                except OSError:
                    # An enclosing job that forbids breakaway. Started inside it
                    # instead, where it survives unless that job ends children.
                    logger.warning("could not start the installer outside the job object")
                    self._launcher(command, close_fds=True, creationflags=flags)
            else:
                self._launcher(command, close_fds=True, start_new_session=True)
            self._status.state = "applying"

    def apply_on_exit(self) -> bool:
        """Install a ready update as the app closes, if updates are automatic."""
        status = self.snapshot()
        if not status.auto or status.state != "ready":
            return False
        try:
            self.apply(relaunch=False)
        except (UpdateError, OSError) as exc:
            logger.warning("could not install the update on exit: %s", exc)
            return False
        return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _acceptable_feed(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.netloc:
        return True
    return parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost", "::1")


def _describe(exc: BaseException) -> str:
    """A sentence for the user, not a traceback."""
    if isinstance(exc, UpdateError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        return f"the update server answered {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "could not reach the update server"
    if isinstance(exc, TimeoutError):
        return "the update server did not answer in time"
    if isinstance(exc, json.JSONDecodeError):
        return "the update server sent something that is not JSON"
    return f"{type(exc).__name__}: {exc}"
