"""Explicit credential verification.

Pasting a key and being told nothing is a bad experience: the failure surfaces
later, during a real meeting, as an error the user cannot connect to the thing
they just did. So verification is a first-class action with a visible result.

Each check uses the **cheapest authenticated call the vendor offers** —
`count_tokens` for Anthropic and `models.list` for OpenAI, both free — because
a "Test" button that quietly spends money is a button people learn not to
press.

The distinction that matters in the result is *invalid* versus *error*. A
rejected key is the user's problem and they can fix it; a network failure or a
vendor outage is not, and telling someone their key is wrong when the internet
is down sends them to regenerate a key that was fine.
"""

from __future__ import annotations

import logging

from koe.providers.credentials import CredentialStore, Status

logger = logging.getLogger(__name__)

VERIFY_TIMEOUT = 15.0


class VerificationResult:
    """Outcome of checking one credential.

    `detail` is English prose, right for a log line or an API consumer. `code`
    is the enumerable version, so a bilingual UI can say the same thing in the
    language its reader chose instead of matching English sentences. Outcomes
    that carry a vendor exception have no code — there is nothing to enumerate
    — and the client shows the detail as-is.
    """

    __slots__ = ("code", "detail", "status")

    def __init__(self, status: Status, detail: str = "", code: str = "") -> None:
        self.status = status
        self.detail = detail
        self.code = code

    @property
    def ok(self) -> bool:
        return self.status is Status.VALID


async def verify(provider: str, store: CredentialStore) -> VerificationResult:
    """Check the configured key for `provider` against the live API."""
    key = store.resolve(provider)
    if not key:
        return VerificationResult(Status.UNKNOWN, "no key is configured", "no_key")

    try:
        if provider == "anthropic":
            return await _verify_anthropic(key)
        if provider == "openai":
            return await _verify_openai(key)
    except ImportError:
        return VerificationResult(
            Status.ERROR,
            f"the {provider} client library is not installed in this build",
            "sdk_missing",
        )
    except Exception as exc:  # noqa: BLE001 - classified below, never propagated
        logger.warning("verification failed for %s: %r", provider, exc)
        return VerificationResult(Status.ERROR, _describe(exc))

    return VerificationResult(Status.ERROR, f"unknown provider {provider!r}", "unknown_provider")


def _describe(exc: BaseException) -> str:
    """A message worth showing a person.

    Vendor exceptions stringify to something that includes the request body,
    which for an auth failure can echo the key back. Only the type and a short
    reason are surfaced.
    """
    text = str(exc)
    if len(text) > 160:
        text = text[:157] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


async def _verify_anthropic(key: str) -> VerificationResult:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=key, timeout=VERIFY_TIMEOUT, max_retries=0)
    try:
        # count_tokens authenticates but generates nothing, so it is free.
        await client.messages.count_tokens(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "ping"}],
        )
    except anthropic.AuthenticationError:
        return VerificationResult(Status.INVALID, "the key was rejected", "rejected")
    except anthropic.PermissionDeniedError:
        return VerificationResult(
            Status.INVALID,
            "the key is valid but lacks permission for this model",
            "no_permission",
        )
    except anthropic.RateLimitError:
        # Rate limiting proves the key authenticated.
        return VerificationResult(
            Status.VALID, "rate limited, but the key is accepted", "rate_limited"
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
        return VerificationResult(
            Status.ERROR, f"could not reach Anthropic: {_describe(exc)}", "unreachable"
        )
    return VerificationResult(Status.VALID, "accepted", "accepted")


async def _verify_openai(key: str) -> VerificationResult:
    import openai

    client = openai.AsyncOpenAI(api_key=key, timeout=VERIFY_TIMEOUT, max_retries=0)
    try:
        await client.models.list()
    except openai.AuthenticationError:
        return VerificationResult(Status.INVALID, "the key was rejected", "rejected")
    except openai.PermissionDeniedError:
        return VerificationResult(
            Status.INVALID, "the key lacks the required permission", "no_permission"
        )
    except openai.RateLimitError:
        return VerificationResult(
            Status.VALID, "rate limited, but the key is accepted", "rate_limited"
        )
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        return VerificationResult(
            Status.ERROR, f"could not reach OpenAI: {_describe(exc)}", "unreachable"
        )
    return VerificationResult(Status.VALID, "accepted", "accepted")
