"""Groundedness checking for generated minutes.

The failure mode this exists for is specific: an LLM asked to summarize a
meeting will occasionally invent an action item nobody agreed to, or attach a
deadline nobody said, and it will do so fluently enough that a human reviewer
skims past it. In a minutes product that invented task lands in someone's
backlog with a due date.

Because the schema requires a verbatim `source_quote` on every claim, checking
is a string operation rather than a second model call. No LLM judge, no
judge-uncertainty to reason about, no additional latency or cost.

Matching runs on **scoring-normalized text**, because a model asked to copy a
span verbatim will still reliably differ in punctuation and width -- and those
differences are exactly what the normalization layer was built to erase. It is
not an exact-match check for the same reason CER is not: the differences that
survive normalization are the ones that mean something.

Support is scored by longest common substring rather than pass/fail, so a
lightly-reworded quote scores 0.9 and a fabricated one scores near 0. That
distinction lets the caller keep the first and drop the second, instead of
throwing away every claim whose citation is imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum

from koe.minutes.schema import ActionItem, Minutes, SourcedClaim
from koe.text.normalize import normalize_for_scoring

#: Below this, a citation is not credible enough to publish.
DEFAULT_SUPPORT_THRESHOLD = 0.85


class Verdict(StrEnum):
    GROUNDED = "grounded"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    UNCITED = "uncited"


@dataclass(slots=True)
class ClaimCheck:
    """The result of checking one claim against the transcript."""

    claim: str
    verdict: Verdict
    support: float
    quote: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.GROUNDED


@dataclass(slots=True)
class GroundednessReport:
    """Every claim in a set of minutes, checked."""

    checks: list[ClaimCheck] = field(default_factory=list)
    unknown_speakers: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def grounded(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def failures(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def score(self) -> float:
        """Fraction of claims that are properly supported."""
        return self.grounded / self.total if self.total else 1.0

    @property
    def passed(self) -> bool:
        return not self.failures and not self.unknown_speakers

    def summary(self) -> str:
        lines = [f"groundedness: {self.grounded}/{self.total} claims supported ({self.score:.0%})"]
        for check in self.failures:
            lines.append(f"  [{check.verdict.value}] {check.claim[:60]!r}: {check.detail}")
        for speaker in self.unknown_speakers:
            lines.append(f"  [unknown-speaker] {speaker!r} is not a participant")
        return "\n".join(lines)


def support_score(quote: str, transcript: str) -> float:
    """How much of `quote` appears in `transcript`, in [0, 1].

    Both sides are scoring-normalized first, so width, punctuation and numeral
    spelling do not count against a citation that is otherwise exact.
    """
    needle = normalize_for_scoring(quote)
    haystack = normalize_for_scoring(transcript)
    if not needle:
        return 0.0
    if not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    match = SequenceMatcher(None, needle, haystack, autojunk=False).find_longest_match(
        0, len(needle), 0, len(haystack)
    )
    return match.size / len(needle)


def check_claim(
    text: str,
    quote: str,
    transcript: str,
    *,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> ClaimCheck:
    """Check a single claim's citation."""
    if not quote.strip():
        return ClaimCheck(
            claim=text,
            verdict=Verdict.UNCITED,
            support=0.0,
            detail="no source_quote provided",
        )

    score = support_score(quote, transcript)
    if score >= threshold:
        verdict = Verdict.GROUNDED
        detail = ""
    elif score >= threshold * 0.6:
        verdict = Verdict.WEAK
        detail = f"only {score:.0%} of the quote appears in the transcript"
    else:
        verdict = Verdict.UNSUPPORTED
        detail = f"quote not found in transcript (support {score:.0%})"
    return ClaimCheck(claim=text, verdict=verdict, support=score, quote=quote, detail=detail)


def _claim_text(claim: SourcedClaim) -> str:
    if isinstance(claim, ActionItem):
        return claim.task
    return getattr(claim, "statement", "")


def check_minutes(
    minutes: Minutes,
    transcript: str,
    *,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    known_speakers: list[str] | None = None,
) -> GroundednessReport:
    """Check every claim in `minutes` against `transcript`.

    Also verifies that action-item owners are real participants. Assigning a
    task to a person who was not in the meeting is a distinct failure from
    inventing the task, and it is the one that causes an awkward conversation.
    """
    report = GroundednessReport()
    for claim in minutes.claims():
        report.checks.append(
            check_claim(_claim_text(claim), claim.source_quote, transcript, threshold=threshold)
        )

    speakers = set(known_speakers or minutes.participants)
    if speakers:
        for item in minutes.action_items:
            if item.owner and item.owner not in speakers:
                report.unknown_speakers.append(item.owner)

    return report


def drop_unsupported(
    minutes: Minutes,
    transcript: str,
    *,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> tuple[Minutes, GroundednessReport]:
    """Return `minutes` with unsupported claims removed.

    Dropping rather than flagging is the right default for a document a person
    will act on: an unsupported action item that is merely marked "unverified"
    still ends up in a task list, and the reader has no way to check it. A weak
    citation is kept -- the claim is real, the quote was just tidied up.

    The returned report describes the **cleaned** minutes, not the input. A
    report that still listed the removed claims as failures would mark a
    document untrustworthy for problems it no longer has; callers that need to
    know what was removed compare the two documents.
    """

    def keep(claim: SourcedClaim) -> bool:
        check = check_claim(_claim_text(claim), claim.source_quote, transcript, threshold=threshold)
        return check.verdict in (Verdict.GROUNDED, Verdict.WEAK)

    cleaned = minutes.model_copy(
        update={
            "decisions": [d for d in minutes.decisions if keep(d)],
            "action_items": [
                a.model_copy(
                    update={"owner": a.owner if a.owner in set(minutes.participants) else ""}
                )
                for a in minutes.action_items
                if keep(a)
            ],
        }
    )
    return (cleaned, check_minutes(cleaned, transcript, threshold=threshold))
