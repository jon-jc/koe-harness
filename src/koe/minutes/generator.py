"""Transcript -> 議事録, with verification and repair.

The generation loop is short but every step earns its place:

1. Build a prompt whose *stable* half (instructions + schema) is cacheable and
   whose *volatile* half (the transcript) comes after it.
2. Ask for schema-constrained JSON, so the shape is guaranteed at the source
   rather than parsed hopefully at this end.
3. Validate against the pydantic model. Constrained decoding guarantees shape,
   not semantics -- a required field can still arrive empty.
4. Check groundedness. This is the step that catches invention.
5. If claims are unsupported, repair **once**, naming the specific offending
   quotes.
6. Drop whatever is still unsupported, and report what was dropped.

Step 5 repairs once rather than looping. A model that cannot cite a claim on
the second attempt is not going to find a citation on the fifth; it will
either fabricate a better-looking quote or churn tokens. One targeted repair
recovers the common case -- a lightly-paraphrased quote -- and the deterministic
drop in step 6 handles the rest without unbounded cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from koe.domain.transcript import Transcript
from koe.minutes.guardrails import (
    DEFAULT_SUPPORT_THRESHOLD,
    GroundednessReport,
    check_minutes,
    drop_unsupported,
)
from koe.minutes.prompts import repair_prompt, system_prompt, user_prompt
from koe.minutes.schema import Minutes, minutes_json_schema
from koe.providers.base import LLMProvider, Message, ProviderError, Usage
from koe.text.script import Language

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MinutesResult:
    """Generated minutes plus everything needed to audit them."""

    minutes: Minutes
    report: GroundednessReport
    usage: Usage = field(default_factory=Usage)
    repairs: int = 0
    dropped: list[str] = field(default_factory=list)
    raw_json: dict[str, Any] | None = None

    @property
    def trustworthy(self) -> bool:
        """Whether every surviving claim is properly cited."""
        return self.report.passed or not self.report.failures

    def audit(self) -> str:
        lines = [self.report.summary()]
        if self.repairs:
            lines.append(f"repairs: {self.repairs}")
        if self.dropped:
            lines.append(f"dropped {len(self.dropped)} unsupported claim(s):")
            lines += [f"  - {d}" for d in self.dropped]
        lines.append(
            f"cost: ${self.usage.cost_usd:.4f} "
            f"({self.usage.input_tokens} in / {self.usage.output_tokens} out"
            f"{', cached' if self.usage.cached else ''})"
        )
        return "\n".join(lines)


class MinutesGenerator:
    """Produces verified 議事録 from a diarized transcript."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
        max_repairs: int = 1,
    ) -> None:
        self.llm = llm
        self.support_threshold = support_threshold
        self.max_repairs = max_repairs

    async def generate(
        self,
        transcript: Transcript,
        *,
        language: Language | None = None,
    ) -> MinutesResult:
        """Generate minutes for `transcript` and verify them against it."""
        lang = language or transcript.dominant_language()
        if lang is Language.UNKNOWN:
            lang = Language.JA

        lines = transcript.transcript_lines()
        body = "\n".join(lines)
        participants = sorted({s.speaker for s in transcript.final_segments if s.speaker})

        if not body.strip():
            # Nothing was said. Returning empty minutes is correct; asking a
            # model to summarize silence is the classic way to get invention.
            return MinutesResult(
                minutes=Minutes(language=lang, participants=participants),
                report=GroundednessReport(),
            )

        system = system_prompt(lang)
        messages = [
            Message(role="user", content=user_prompt(body, lang, participants=participants))
        ]

        minutes, usage, raw = await self._request(messages, system=system, language=lang)
        total_usage = usage

        report = check_minutes(
            minutes, body, threshold=self.support_threshold, known_speakers=participants
        )
        repairs = 0

        if report.failures and self.max_repairs > 0:
            problems = [f"{c.claim} (quote: {c.quote[:60]!r})" for c in report.failures]
            logger.info("repairing %d unsupported claim(s)", len(problems))
            messages = [
                *messages,
                Message(role="assistant", content=minutes.model_dump_json()),
                Message(role="user", content=repair_prompt(problems, lang)),
            ]
            try:
                repaired, repair_usage, raw = await self._request(
                    messages, system=system, language=lang
                )
            except ProviderError as exc:
                # A failed repair is not a failed generation: the deterministic
                # drop below still produces trustworthy minutes.
                logger.warning("repair attempt failed, falling back to dropping: %s", exc)
            else:
                minutes = repaired
                total_usage = total_usage + repair_usage
                repairs = 1
                report = check_minutes(
                    minutes, body, threshold=self.support_threshold, known_speakers=participants
                )

        before = {c.claim for c in report.checks}
        cleaned, final_report = drop_unsupported(minutes, body, threshold=self.support_threshold)
        after = {
            *(d.statement for d in cleaned.decisions),
            *(a.task for a in cleaned.action_items),
        }
        dropped = sorted(before - after)

        cleaned = cleaned.model_copy(
            update={
                "language": lang,
                "participants": cleaned.participants or participants,
            }
        )

        return MinutesResult(
            minutes=cleaned,
            report=final_report,
            usage=total_usage,
            repairs=repairs,
            dropped=dropped,
            raw_json=raw,
        )

    async def _request(
        self,
        messages: list[Message],
        *,
        system: str,
        language: Language,
    ) -> tuple[Minutes, Usage, dict[str, Any] | None]:
        """One schema-constrained call, validated into a :class:`Minutes`."""
        schema = minutes_json_schema()

        complete_json = getattr(self.llm, "complete_json", None)
        if complete_json is not None:
            payload, response = await complete_json(messages, schema=schema, system=system)
        else:
            # Providers without constrained decoding still have to work; the
            # guardrails do not depend on how the JSON was produced.
            response = await self.llm.complete(messages, system=system)
            payload = _loads(response.text, provider=response.usage.provider)

        try:
            minutes = Minutes.model_validate(payload)
        except ValidationError as exc:
            raise ProviderError(
                f"model returned JSON that does not match the minutes schema: {exc}",
                provider=response.usage.provider,
                retryable=False,
            ) from exc

        return (minutes, response.usage, payload)


def _loads(text: str, *, provider: str) -> dict[str, Any]:
    import json

    stripped = text.strip()
    # Models still occasionally wrap JSON in a markdown fence despite being
    # asked not to; stripping it is cheaper than a repair round trip.
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"model returned unparseable JSON: {exc}", provider=provider, retryable=False
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderError(
            f"model returned {type(parsed).__name__}, expected an object",
            provider=provider,
            retryable=False,
        )
    return parsed
