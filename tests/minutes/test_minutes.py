"""議事録 generation: schema, groundedness, repair, and dropping."""

from __future__ import annotations

import json

import pytest

from koe.domain.transcript import Segment, Transcript
from koe.minutes.generator import MinutesGenerator
from koe.minutes.guardrails import (
    Verdict,
    check_claim,
    check_minutes,
    drop_unsupported,
    support_score,
)
from koe.minutes.schema import ActionItem, Decision, Minutes, minutes_json_schema
from koe.providers.base import ProviderError
from koe.providers.mock import MockLLM
from koe.text.script import Language

TRANSCRIPT_JA = """田中: 本日の議題は第三四半期の売上レビューです。
鈴木: 売上は前年比で百二十パーセント、目標を達成しました。
田中: KPIのダッシュボードは来週までに更新できますか。
佐藤: はい、金曜日までに対応します。
鈴木: 新機能のリリースは三月十日を予定しています。"""


def transcript_from(text: str) -> Transcript:
    segments = []
    for i, line in enumerate(text.strip().splitlines()):
        speaker, _, content = line.partition(": ")
        segments.append(
            Segment(
                text=content,
                start=float(i * 4),
                end=float(i * 4 + 3),
                speaker=speaker,
                language=Language.JA,
            )
        )
    return Transcript(segments=segments, language=Language.JA)


def minutes_payload(**overrides: object) -> dict:
    payload = {
        "language": "ja",
        "title": "第三四半期 売上レビュー",
        "participants": ["田中", "鈴木", "佐藤"],
        "summary": "第三四半期の売上をレビューしました。",
        "topics": [{"title": "売上", "summary": "目標を達成しました。"}],
        "decisions": [
            {
                "statement": "新機能のリリースは三月十日とする",
                "source_quote": "新機能のリリースは三月十日を予定しています。",
                "speaker": "鈴木",
            }
        ],
        "action_items": [
            {
                "task": "KPIダッシュボードを更新する",
                "owner": "佐藤",
                "due": "金曜日",
                "source_quote": "はい、金曜日までに対応します。",
                "speaker": "佐藤",
            }
        ],
    }
    payload.update(overrides)  # type: ignore[arg-type]
    return payload


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_schema_forbids_extra_properties() -> None:
    """Extra keys validate but carry content nothing reads."""
    schema = minutes_json_schema()
    assert schema["additionalProperties"] is False
    for definition in schema.get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False


def test_every_claim_type_requires_a_quote_field() -> None:
    assert "source_quote" in Decision.model_fields
    assert "source_quote" in ActionItem.model_fields


def test_render_produces_japanese_minutes() -> None:
    minutes = Minutes.model_validate(minutes_payload())
    rendered = minutes.render()
    assert "出席者" in rendered
    assert "決定事項" in rendered
    assert "アクションアイテム" in rendered
    assert "担当: 佐藤" in rendered


def test_render_produces_english_minutes() -> None:
    minutes = Minutes(
        language=Language.EN,
        title="Q3 review",
        participants=["Alice"],
        action_items=[ActionItem(task="Update the dashboard", owner="Alice", due="Friday")],
    )
    rendered = minutes.render()
    assert "Participants" in rendered
    assert "Action items" in rendered
    assert "Owner: Alice" in rendered


def test_unassigned_owner_renders_without_inventing_one() -> None:
    minutes = Minutes(
        language=Language.JA, action_items=[ActionItem(task="対応する", owner="", due="")]
    )
    assert "未割当" in minutes.render()


# --------------------------------------------------------------------------
# groundedness
# --------------------------------------------------------------------------


def test_a_verbatim_quote_is_fully_supported() -> None:
    assert support_score("金曜日までに対応します", TRANSCRIPT_JA) == 1.0


def test_normalization_makes_citation_robust_to_formatting() -> None:
    """A model copying verbatim still differs in punctuation and width."""
    assert support_score("金曜日までに対応します。", TRANSCRIPT_JA) == 1.0
    assert support_score("ＫＰＩのダッシュボード", TRANSCRIPT_JA) == 1.0


def test_an_invented_quote_scores_near_zero() -> None:
    score = support_score("来月中に全社展開することで合意しました", TRANSCRIPT_JA)
    assert score < 0.5


def test_a_missing_quote_is_uncited() -> None:
    check = check_claim("something", "", TRANSCRIPT_JA)
    assert check.verdict is Verdict.UNCITED
    assert not check.ok


def test_a_fabricated_claim_is_unsupported() -> None:
    check = check_claim("予算を倍増する", "予算を倍増することで合意しました", TRANSCRIPT_JA)
    assert check.verdict is Verdict.UNSUPPORTED


def test_a_grounded_claim_passes() -> None:
    check = check_claim("金曜までに対応", "金曜日までに対応します", TRANSCRIPT_JA)
    assert check.verdict is Verdict.GROUNDED
    assert check.ok


def test_report_scores_the_whole_document() -> None:
    minutes = Minutes.model_validate(minutes_payload())
    report = check_minutes(minutes, TRANSCRIPT_JA)
    assert report.total == 2
    assert report.grounded == 2
    assert report.score == 1.0
    assert report.passed


def test_an_owner_who_was_not_in_the_meeting_is_flagged() -> None:
    """A distinct failure from inventing the task, and the more awkward one."""
    minutes = Minutes.model_validate(
        minutes_payload(
            action_items=[
                {
                    "task": "対応する",
                    "owner": "山田",  # never spoke
                    "source_quote": "はい、金曜日までに対応します。",
                }
            ]
        )
    )
    report = check_minutes(minutes, TRANSCRIPT_JA, known_speakers=["田中", "鈴木", "佐藤"])
    assert report.unknown_speakers == ["山田"]
    assert not report.passed


def test_unsupported_claims_are_dropped_not_flagged() -> None:
    """An 'unverified' action item still lands in someone's task list."""
    minutes = Minutes.model_validate(
        minutes_payload(
            action_items=[
                {
                    "task": "実在するタスク",
                    "owner": "佐藤",
                    "source_quote": "はい、金曜日までに対応します。",
                },
                {
                    "task": "捏造されたタスク",
                    "owner": "佐藤",
                    "source_quote": "来月までに全社展開してください",
                },
            ]
        )
    )

    cleaned, _ = drop_unsupported(minutes, TRANSCRIPT_JA)

    tasks = [a.task for a in cleaned.action_items]
    assert tasks == ["実在するタスク"]


def test_a_weakly_cited_claim_survives() -> None:
    """The claim is real; the quote was just tidied up."""
    minutes = Minutes(
        language=Language.JA,
        participants=["佐藤"],
        action_items=[
            ActionItem(
                task="対応する",
                owner="佐藤",
                source_quote="はい、金曜日までに対応しますとのことでした",
            )
        ],
    )
    cleaned, _ = drop_unsupported(minutes, TRANSCRIPT_JA)
    assert len(cleaned.action_items) == 1


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


async def test_clean_generation_produces_verified_minutes() -> None:
    llm = MockLLM(default_response=json.dumps(minutes_payload(), ensure_ascii=False))
    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))

    assert result.trustworthy
    assert result.repairs == 0
    assert not result.dropped
    assert len(result.minutes.action_items) == 1
    assert result.minutes.action_items[0].owner == "佐藤"


async def test_an_invented_action_item_triggers_repair_then_survives_or_is_dropped() -> None:
    bad = minutes_payload(
        action_items=[
            {
                "task": "全社展開する",
                "owner": "佐藤",
                "due": "来月",
                "source_quote": "来月までに全社展開してください",  # never said
            }
        ]
    )
    good = minutes_payload()

    class Sequenced(MockLLM):
        """First call invents, second call is corrected."""

        async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            payload = bad if self.calls == 1 else good
            self.default_response = json.dumps(payload, ensure_ascii=False)
            return await MockLLM.complete(self, messages, **kwargs)

    result = await MinutesGenerator(Sequenced()).generate(transcript_from(TRANSCRIPT_JA))

    assert result.repairs == 1
    assert result.trustworthy
    assert [a.task for a in result.minutes.action_items] == ["KPIダッシュボードを更新する"]


async def test_an_unrepairable_claim_is_dropped() -> None:
    bad = minutes_payload(
        action_items=[
            {
                "task": "捏造タスク",
                "owner": "佐藤",
                "source_quote": "この文は文字起こしに存在しません",
            }
        ]
    )
    llm = MockLLM(default_response=json.dumps(bad, ensure_ascii=False))

    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))

    assert result.repairs == 1  # tried once
    assert result.minutes.action_items == []
    assert "捏造タスク" in result.dropped
    assert result.trustworthy  # what remains is trustworthy


async def test_a_failed_repair_still_yields_trustworthy_minutes() -> None:
    """A failed repair is not a failed generation."""
    bad = minutes_payload(
        action_items=[{"task": "捏造", "owner": "佐藤", "source_quote": "存在しない引用"}]
    )

    class FailsOnRepair(MockLLM):
        async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            if self.calls >= 1:
                self.calls += 1
                raise ProviderError("upstream down", provider="mock-llm")
            return await MockLLM.complete(self, messages, **kwargs)

    llm = FailsOnRepair(default_response=json.dumps(bad, ensure_ascii=False))
    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))

    assert result.minutes.action_items == []
    assert result.dropped


async def test_an_empty_transcript_produces_empty_minutes_without_calling_the_model() -> None:
    """Asking a model to summarize silence is how you get invention."""
    llm = MockLLM(default_response="{}")
    result = await MinutesGenerator(llm).generate(Transcript(segments=[]))

    assert llm.calls == 0
    assert result.minutes.action_items == []
    assert result.trustworthy


async def test_malformed_json_is_a_non_retryable_error() -> None:
    llm = MockLLM(default_response="this is not json")
    with pytest.raises(ProviderError) as excinfo:
        await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))
    assert not excinfo.value.retryable


async def test_json_wrapped_in_a_markdown_fence_is_recovered() -> None:
    fenced = "```json\n" + json.dumps(minutes_payload(), ensure_ascii=False) + "\n```"
    llm = MockLLM(default_response=fenced)
    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))
    assert result.minutes.title


async def test_schema_violations_are_non_retryable() -> None:
    llm = MockLLM(default_response=json.dumps({"decisions": "not a list"}))
    with pytest.raises(ProviderError, match="minutes schema"):
        await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))


async def test_audit_reports_cost_and_drops() -> None:
    bad = minutes_payload(
        action_items=[{"task": "捏造", "owner": "佐藤", "source_quote": "存在しない"}]
    )
    llm = MockLLM(default_response=json.dumps(bad, ensure_ascii=False))
    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))

    audit = result.audit()
    assert "groundedness" in audit
    assert "dropped" in audit
    assert "cost" in audit


async def test_participants_are_taken_from_the_transcript() -> None:
    llm = MockLLM(default_response=json.dumps(minutes_payload(participants=[]), ensure_ascii=False))
    result = await MinutesGenerator(llm).generate(transcript_from(TRANSCRIPT_JA))
    assert set(result.minutes.participants) == {"田中", "鈴木", "佐藤"}
