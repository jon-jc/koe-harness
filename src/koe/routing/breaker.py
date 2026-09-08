"""Circuit breakers, so a failing provider stops being asked.

When a backend starts failing, continuing to send it traffic costs latency on
every request before the fallback runs. On a realtime path that is the whole
budget: a 3-second timeout against a dead vendor means the user waits 3 seconds
to receive a transcript from the *second* provider, on every utterance.

The breaker exists to make failure fast. After enough consecutive failures it
opens and the provider is skipped without being called. After a cooldown it
half-opens and lets a single probe through -- if that succeeds the provider is
restored, if it fails the cooldown restarts.

Two properties worth stating:

* **Only retryable failures count.** A malformed-request error means *this
  request* is broken, not the provider. Counting it would open the breaker on a
  backend that is perfectly healthy and take out a working dependency.
* **Half-open admits exactly one probe.** Letting the full flood through on
  recovery is how a service that just came back gets knocked over again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"  # healthy, traffic flows
    OPEN = "open"  # failing, traffic skipped
    HALF_OPEN = "half_open"  # probing recovery


@dataclass(slots=True)
class CircuitBreaker:
    """Per-provider health tracking."""

    name: str = "provider"
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    success_threshold: int = 1

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _consecutive_successes: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _probe_in_flight: bool = field(default=False, init=False)
    total_failures: int = field(default=0, init=False)
    total_successes: int = field(default=0, init=False)
    total_rejected: int = field(default=0, init=False)

    #: Injectable clock, so tests do not have to sleep through cooldowns.
    clock: object = field(default=None, repr=False)

    def _now(self) -> float:
        return self.clock() if callable(self.clock) else time.monotonic()

    @property
    def state(self) -> BreakerState:
        """Current state, transitioning OPEN -> HALF_OPEN once cooled down."""
        if self._state is BreakerState.OPEN and (
            self._now() - self._opened_at >= self.cooldown_seconds
        ):
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
            logger.info("breaker %s half-open, probing", self.name)
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether a request may be sent to this provider now."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN:
            # exactly one probe at a time; a recovering service must not be
            # hit with the full backlog the moment it responds
            return not self._probe_in_flight
        return False

    def acquire(self) -> bool:
        """Reserve a slot; returns False when the breaker rejects the call."""
        if not self.is_available:
            self.total_rejected += 1
            return False
        if self.state is BreakerState.HALF_OPEN:
            self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self.total_successes += 1
        self._consecutive_failures = 0
        self._probe_in_flight = False
        if self._state is BreakerState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.success_threshold:
                self._state = BreakerState.CLOSED
                self._consecutive_successes = 0
                logger.info("breaker %s closed, provider recovered", self.name)

    def record_failure(self, *, retryable: bool = True) -> None:
        """Record a failure.

        Non-retryable failures are counted for observability but never open the
        breaker: a malformed request says nothing about provider health, and
        opening on it would remove a working backend from rotation.
        """
        self.total_failures += 1
        self._probe_in_flight = False
        if not retryable:
            return

        self._consecutive_failures += 1
        if self._state is BreakerState.HALF_OPEN:
            self._state = BreakerState.OPEN
            self._opened_at = self._now()
            logger.warning("breaker %s reopened, probe failed", self.name)
            return
        if self._consecutive_failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = self._now()
            self._consecutive_successes = 0
            logger.warning(
                "breaker %s opened after %d consecutive failures",
                self.name,
                self._consecutive_failures,
            )

    def reset(self) -> None:
        """Force closed, for operator intervention."""
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._probe_in_flight = False

    def snapshot(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
        }
