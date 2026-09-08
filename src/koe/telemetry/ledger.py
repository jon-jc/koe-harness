"""Cost accounting per session, tenant and model.

A multi-model voice product has a cost structure that is genuinely hard to
reason about from a monthly invoice: ASR bills per audio minute, the LLM bills
per token, both have a cheap and an expensive tier, and the mix shifts with
whatever the router decided. By the time it shows up as one number on a bill,
the question "which of these is expensive and why" is unanswerable.

The ledger records every provider call as it happens, keyed by session and
model, so the answer is available *before* the invoice -- and the unit it
reports in is **cost per hour of audio**, because that is the number that
divides into revenue. Total spend tells you nothing without knowing how much
audio produced it.

It also enforces budgets. A runaway session -- a stuck retry loop, an
accidentally unbounded meeting -- is the failure mode that turns a rounding
error into a real bill, and it is much cheaper to stop it than to discover it
at month end.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from koe.providers.base import Usage

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """A session or tenant exceeded its spending limit."""

    def __init__(self, scope: str, spent: float, limit: float) -> None:
        self.scope = scope
        self.spent = spent
        self.limit = limit
        super().__init__(f"{scope} spent ${spent:.4f}, exceeding its ${limit:.4f} limit")


@dataclass(slots=True)
class Entry:
    """One provider call."""

    session_id: str
    provider: str
    model: str
    modality: str
    cost_usd: float
    audio_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    at: float = field(default_factory=time.time)
    tenant: str = "default"


@dataclass(slots=True)
class Totals:
    """Aggregated spend for one grouping."""

    cost_usd: float = 0.0
    audio_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def cost_per_audio_hour(self) -> float:
        """The unit that divides into revenue."""
        hours = self.audio_seconds / 3600.0
        return self.cost_usd / hours if hours > 0 else 0.0

    def add(self, entry: Entry) -> None:
        self.cost_usd += entry.cost_usd
        self.audio_seconds += entry.audio_seconds
        self.input_tokens += entry.input_tokens
        self.output_tokens += entry.output_tokens
        self.calls += 1


class CostLedger:
    """Records provider spend and enforces budgets.

    Thread-safe: the realtime path records from session tasks while the metrics
    exporter reads totals, and a torn read during aggregation would produce
    numbers that never existed.
    """

    def __init__(
        self,
        *,
        session_limit_usd: float | None = None,
        tenant_limit_usd: float | None = None,
        max_entries: int = 100_000,
    ) -> None:
        self._entries: list[Entry] = []
        self._by_session: dict[str, Totals] = defaultdict(Totals)
        self._by_model: dict[str, Totals] = defaultdict(Totals)
        self._by_tenant: dict[str, Totals] = defaultdict(Totals)
        self._total = Totals()
        self._session_limit = session_limit_usd
        self._tenant_limit = tenant_limit_usd
        self._max_entries = max_entries
        self._lock = threading.Lock()

    # -- recording -----------------------------------------------------------

    def record(
        self,
        usage: Usage,
        *,
        session_id: str = "",
        modality: str = "asr",
        tenant: str = "default",
    ) -> Entry:
        """Record one provider call.

        Raises :class:`BudgetExceeded` when this call takes a scope past its
        limit. The call is still recorded first -- money that was already spent
        does not stop being spent because it broke a limit, and a ledger that
        drops the offending entry under-reports exactly the incident you most
        need to investigate.
        """
        entry = Entry(
            session_id=session_id,
            provider=usage.provider,
            model=usage.model,
            modality=modality,
            cost_usd=usage.cost_usd,
            audio_seconds=usage.audio_seconds,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
            tenant=tenant,
        )

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                # Bounded: totals are authoritative, the entry list is a recent
                # detail buffer. An unbounded list is a slow memory leak in a
                # long-lived server.
                del self._entries[: len(self._entries) - self._max_entries]

            self._total.add(entry)
            self._by_model[f"{usage.provider}/{usage.model}"].add(entry)
            if session_id:
                self._by_session[session_id].add(entry)
            self._by_tenant[tenant].add(entry)

            session_spend = self._by_session[session_id].cost_usd if session_id else 0.0
            tenant_spend = self._by_tenant[tenant].cost_usd

        if self._session_limit is not None and session_spend > self._session_limit:
            logger.error(
                "session %s exceeded its budget: $%.4f > $%.4f",
                session_id,
                session_spend,
                self._session_limit,
            )
            raise BudgetExceeded(f"session {session_id}", session_spend, self._session_limit)

        if self._tenant_limit is not None and tenant_spend > self._tenant_limit:
            logger.error(
                "tenant %s exceeded its budget: $%.4f > $%.4f",
                tenant,
                tenant_spend,
                self._tenant_limit,
            )
            raise BudgetExceeded(f"tenant {tenant}", tenant_spend, self._tenant_limit)

        return entry

    # -- reads ---------------------------------------------------------------

    @property
    def total(self) -> Totals:
        with self._lock:
            return Totals(
                cost_usd=self._total.cost_usd,
                audio_seconds=self._total.audio_seconds,
                input_tokens=self._total.input_tokens,
                output_tokens=self._total.output_tokens,
                calls=self._total.calls,
            )

    def by_model(self) -> dict[str, Totals]:
        with self._lock:
            return dict(self._by_model)

    def by_session(self, session_id: str) -> Totals:
        with self._lock:
            return self._by_session.get(session_id, Totals())

    def by_tenant(self, tenant: str = "default") -> Totals:
        with self._lock:
            return self._by_tenant.get(tenant, Totals())

    def recent(self, limit: int = 50) -> list[Entry]:
        with self._lock:
            return list(self._entries[-limit:])

    def report(self) -> str:
        """Human-readable breakdown, ordered by what costs the most."""
        total = self.total
        lines = [
            f"total: ${total.cost_usd:.4f} over {total.audio_seconds / 60:.1f} min of audio "
            f"(${total.cost_per_audio_hour:.4f}/audio-hour, {total.calls} calls)",
            "",
            f"{'model':<32} {'calls':>6} {'cost':>10} {'$/audio-hr':>12}",
        ]
        for name, totals in sorted(
            self.by_model().items(), key=lambda kv: kv[1].cost_usd, reverse=True
        ):
            lines.append(
                f"{name:<32} {totals.calls:>6} ${totals.cost_usd:>9.4f} "
                f"${totals.cost_per_audio_hour:>11.4f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._by_session.clear()
            self._by_model.clear()
            self._by_tenant.clear()
            self._total = Totals()
