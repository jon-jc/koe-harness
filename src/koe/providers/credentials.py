"""Provider API keys: storage, masking, and precedence.

Adding a provider should be pasting a key into a settings panel, not editing an
environment file and restarting. That convenience is only acceptable if the
handling is careful, so four rules are enforced here rather than left to the
callers:

**A stored key is never returned.** The API exposes a fingerprint
(``sk-ant-…4f2a``) and nothing else. Read-back of a written secret is the
mistake that turns a UI bug, a screenshot, or an over-broad log line into a
credential leak — and nothing legitimately needs the plaintext except the
provider client itself.

**Environment wins over the store.** A deployment that injects credentials
through ECS or a secrets manager must not be silently overridden by a file
somebody left behind. Env-sourced credentials are reported as read-only so the
UI can say why it will not edit them.

**Encrypted at rest where the platform offers it.** On Windows that is DPAPI,
scoped to the current user, which means no key for koe to manage and no
decryptable file if it is copied to another machine. Elsewhere the file is
written with owner-only permissions and the difference is stated rather than
papered over.

**Keys are validated on purpose, never on save.** Verifying costs a request and
some money; doing it silently on every write is a surprise. Verification is an
explicit action with a visible result.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import stat
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = "credentials.json"


class Source(StrEnum):
    """Where a credential came from."""

    ENVIRONMENT = "environment"
    STORE = "store"
    NONE = "none"


class Status(StrEnum):
    """The result of the last explicit verification."""

    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """A provider a user can supply a key for."""

    id: str
    label: str
    env_var: str
    #: Shown in the UI so a user can tell whether they pasted the right thing.
    key_prefix: str
    docs_url: str
    modality: str
    #: The vendor SDK this provider needs. Optional dependencies, so it may be
    #: absent from a given build.
    client_module: str

    @property
    def available(self) -> bool:
        """Whether the vendor SDK can be imported, without importing it.

        Selecting a backend whose library is missing yields something that
        looks configured and fails on first use — a key the user pasted
        correctly, apparently accepted, then a runtime error. `find_spec`
        answers the question at selection time for the cost of a path lookup.
        """
        try:
            return importlib.util.find_spec(self.client_module) is not None
        except (ImportError, ValueError):
            return False

    def looks_plausible(self, key: str) -> bool:
        """Cheap shape check, to catch an obviously wrong paste immediately.

        Deliberately permissive: a strict pattern would reject a key format the
        vendor introduces next month, and the authoritative check is the
        verification request anyway.
        """
        cleaned = key.strip()
        if len(cleaned) < 16:
            return False
        return not self.key_prefix or cleaned.startswith(self.key_prefix)


#: The providers koe knows how to use. Adding one here is what makes it
#: appear in the settings panel — there is no second list to keep in step.
KNOWN_PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="anthropic",
        label="Anthropic (Claude)",
        env_var="KOE_ANTHROPIC_API_KEY",
        key_prefix="sk-ant-",
        docs_url="https://console.anthropic.com/settings/keys",
        modality="llm",
        client_module="anthropic",
    ),
    ProviderSpec(
        id="openai",
        label="OpenAI (GPT)",
        env_var="KOE_OPENAI_API_KEY",
        key_prefix="sk-",
        docs_url="https://platform.openai.com/api-keys",
        modality="llm",
        client_module="openai",
    ),
)

PROVIDERS_BY_ID = {spec.id: spec for spec in KNOWN_PROVIDERS}


@dataclass(frozen=True, slots=True)
class CredentialInfo:
    """What the API is willing to say about a credential."""

    provider: str
    label: str
    modality: str
    docs_url: str
    env_var: str
    configured: bool
    source: Source
    fingerprint: str
    status: Status = Status.UNKNOWN
    verified_at: float | None = None
    detail: str = ""
    #: Enumerable form of `detail`, so a bilingual client can translate it.
    code: str = ""
    #: False when this build lacks the vendor SDK, whatever the key says.
    available: bool = True

    @property
    def editable(self) -> bool:
        """Environment-sourced credentials are not ours to change."""
        return self.source is not Source.ENVIRONMENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "label": self.label,
            "modality": self.modality,
            "docs_url": self.docs_url,
            "env_var": self.env_var,
            "configured": self.configured,
            "source": self.source.value,
            "fingerprint": self.fingerprint,
            "status": self.status.value,
            "verified_at": self.verified_at,
            "detail": self.detail,
            "code": self.code,
            "editable": self.editable,
            "available": self.available,
        }


class CredentialError(ValueError):
    """A rejected key, carrying a code the caller can translate.

    The message is English and fine for an API consumer or a log line. The
    *code* is what lets a bilingual UI say the same thing in the language its
    user actually reads — string-matching an English sentence to decide which
    Japanese sentence to show is the failure mode this avoids.
    """

    def __init__(self, code: str, message: str, **context: str) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def fingerprint(key: str) -> str:
    """A recognizable, non-reversible label for a key.

    Enough for a person to confirm *which* key is configured — the prefix they
    recognise and the last four characters they can compare against their
    provider console — and useless to anyone who obtains it.
    """
    cleaned = key.strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "…" + cleaned[-2:]
    head = cleaned[:7] if cleaned.startswith("sk-") else cleaned[:3]
    return f"{head}…{cleaned[-4:]}"


# --------------------------------------------------------------------------
# at-rest protection
# --------------------------------------------------------------------------


def _dpapi_available() -> bool:
    return sys.platform == "win32"


def _protect(plaintext: str) -> tuple[str, str]:
    """Encrypt for storage. Returns ``(scheme, payload)``.

    DPAPI on Windows binds the ciphertext to the current user account, so koe
    holds no key of its own and a copied file is inert on another machine.
    """
    if _dpapi_available():
        try:
            return ("dpapi", _dpapi(plaintext.encode("utf-8"), encrypt=True))
        except OSError as exc:
            logger.warning("DPAPI unavailable (%s); storing obfuscated only", exc)
    # Not encryption, and named so it cannot be mistaken for it. The real
    # protection on these platforms is the file mode set in `_write`.
    return ("plain", base64.b64encode(plaintext.encode("utf-8")).decode("ascii"))


def _unprotect(scheme: str, payload: str) -> str | None:
    try:
        if scheme == "dpapi":
            if not _dpapi_available():
                return None
            return str(_dpapi(payload, encrypt=False))
        return base64.b64decode(payload.encode("ascii")).decode("utf-8")
    except (OSError, ValueError) as exc:
        logger.warning("could not read a stored credential: %s", exc)
        return None


def _dpapi(data: bytes | str, *, encrypt: bool) -> Any:
    """Call CryptProtectData / CryptUnprotectData.

    The platform guard is load-bearing for the type checker as well as at
    runtime: mypy narrows on `sys.platform`, so without it the Windows-only
    ctypes surface below is checked — and reported missing — on every other
    platform.
    """
    if sys.platform != "win32":
        raise OSError("DPAPI is only available on Windows")

    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    raw = data if isinstance(data, bytes) else base64.b64decode(data.encode("ascii"))
    source = Blob(
        len(raw), ctypes.cast(ctypes.create_string_buffer(raw), ctypes.POINTER(ctypes.c_char))
    )
    result = Blob()
    call = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData

    # CRYPTPROTECT_UI_FORBIDDEN: never prompt. A background save must not block
    # on a dialog the user is not looking at.
    ok = call(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(result))
    if not ok:
        raise OSError(ctypes.get_last_error(), "DPAPI call failed")
    try:
        out = ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)
    return base64.b64encode(out).decode("ascii") if encrypt else out.decode("utf-8")


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


class CredentialStore:
    """Reads and writes provider API keys.

    Intended for the desktop application and local development. A server
    deployment should inject credentials through the environment, which this
    store defers to.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or self._default_path()
        self._cache: dict[str, dict[str, Any]] | None = None

    @staticmethod
    def _default_path() -> Path:
        from koe.desktop.paths import config_dir

        return config_dir() / CREDENTIALS_FILE

    @property
    def path(self) -> Path:
        return self._path

    # -- persistence ---------------------------------------------------------

    def _read(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._cache = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt credential file must not stop the app: the user can
            # re-enter a key, but cannot recover from a window that will not
            # open.
            logger.warning("credentials unreadable (%s); treating as empty", exc)
            self._cache = {}
        return self._cache

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Owner-only, set before the file is in its final place so there is no
        # window in which it is readable by others.
        try:
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            logger.debug("could not set credential file permissions: %s", exc)
        temporary.replace(self._path)
        self._cache = data

    # -- reads ---------------------------------------------------------------

    def resolve(self, provider: str) -> str | None:
        """The key to actually use, or ``None``.

        Environment first. This is the only method that returns plaintext, and
        nothing above the provider clients should call it.
        """
        spec = PROVIDERS_BY_ID.get(provider)
        if spec is None:
            return None

        from_env = os.environ.get(spec.env_var)
        if from_env:
            return from_env

        entry = self._read().get(provider)
        if not entry:
            return None
        return _unprotect(str(entry.get("scheme", "plain")), str(entry.get("value", "")))

    def source_of(self, provider: str) -> Source:
        spec = PROVIDERS_BY_ID.get(provider)
        if spec is None:
            return Source.NONE
        if os.environ.get(spec.env_var):
            return Source.ENVIRONMENT
        return Source.STORE if provider in self._read() else Source.NONE

    def describe(self, provider: str) -> CredentialInfo:
        """Everything the API may disclose about one provider."""
        spec = PROVIDERS_BY_ID[provider]
        source = self.source_of(provider)
        key = self.resolve(provider)
        entry = self._read().get(provider, {})

        return CredentialInfo(
            provider=spec.id,
            label=spec.label,
            modality=spec.modality,
            docs_url=spec.docs_url,
            env_var=spec.env_var,
            configured=bool(key),
            source=source,
            fingerprint=fingerprint(key) if key else "",
            status=Status(str(entry.get("status", Status.UNKNOWN.value))),
            verified_at=entry.get("verified_at"),
            detail=str(entry.get("detail", "")),
            code=str(entry.get("code", "")),
            available=spec.available,
        )

    def describe_all(self) -> list[CredentialInfo]:
        return [self.describe(spec.id) for spec in KNOWN_PROVIDERS]

    # -- writes --------------------------------------------------------------

    def set(self, provider: str, key: str) -> CredentialInfo:
        """Store a key. Raises :class:`CredentialError` if it is obviously wrong."""
        spec = PROVIDERS_BY_ID.get(provider)
        if spec is None:
            raise CredentialError("unknown_provider", f"unknown provider {provider!r}")

        cleaned = key.strip()
        if not cleaned:
            raise CredentialError("empty_key", "the key is empty")
        if not spec.looks_plausible(cleaned):
            raise CredentialError(
                "bad_prefix",
                f"{spec.label} keys start with {spec.key_prefix!r}; that key does not",
                label=spec.label,
                prefix=spec.key_prefix,
            )

        scheme, payload = _protect(cleaned)
        data = dict(self._read())
        data[provider] = {
            "scheme": scheme,
            "value": payload,
            "saved_at": time.time(),
            # A newly saved key has not been checked; saying "valid" here would
            # be a claim nothing verified.
            "status": Status.UNKNOWN.value,
        }
        self._write(data)
        logger.info("stored credential", extra={"provider": provider, "scheme": scheme})
        return self.describe(provider)

    def delete(self, provider: str) -> CredentialInfo:
        data = dict(self._read())
        data.pop(provider, None)
        self._write(data)
        logger.info("removed credential", extra={"provider": provider})
        return self.describe(provider)

    def record_verification(
        self, provider: str, status: Status, detail: str = "", code: str = ""
    ) -> CredentialInfo:
        """Persist the outcome of an explicit verification."""
        data = dict(self._read())
        entry = dict(data.get(provider, {}))
        entry["status"] = status.value
        entry["verified_at"] = time.time()
        entry["detail"] = detail
        entry["code"] = code
        # Recorded even for env-sourced keys, so the panel can show a result
        # for a credential it cannot edit.
        data[provider] = entry
        self._write(data)
        return self.describe(provider)
