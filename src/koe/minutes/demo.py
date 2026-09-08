"""Canned 議事録 for the demo paths.

Used by both ``koe minutes`` and the web demo, so the two show the same thing
and there is one place to change it.

**The canned output deliberately contains one fabricated decision.** That is
not a shortcut around writing a better fixture — it is the point of the demo.
The guardrail rejecting an invented claim is the most important behaviour in
the LLM layer, and a real model cannot be relied upon to hallucinate on cue, so
demonstrating it requires planting one. The dropped claim is labelled as
deliberate wherever it is surfaced, so nobody mistakes the demo for a real
model failing.
"""

from __future__ import annotations

from koe.domain.transcript import Transcript
from koe.minutes.schema import ActionItem, Decision, Minutes, Topic
from koe.providers.mock import MockLLM
from koe.text.script import Language

#: The claim planted to be caught. Surfaced so callers can label it.
FABRICATED_DECISION = "来月中に全社展開を完了する"


def canned_minutes(transcript: Transcript | None = None) -> Minutes:
    """A plausible 議事録 for the bundled quarterly meeting."""
    participants = (
        sorted({s.speaker for s in transcript.final_segments if s.speaker})
        if transcript is not None
        else ["佐藤", "田中", "鈴木"]
    )
    language = transcript.dominant_language() if transcript is not None else Language.JA
    if language is Language.UNKNOWN:
        language = Language.JA

    return Minutes(
        language=language,
        title="第三四半期 売上レビュー",
        participants=participants,
        summary=(
            "第三四半期の売上は前年比120パーセントで目標を達成しました。"
            "KPIダッシュボードの更新期限と、新機能のリリース日を確認しました。"
        ),
        topics=[
            Topic(title="売上レビュー", summary="前年比120パーセントで目標を達成しました。"),
            Topic(title="リリース計画", summary="新機能のリリース日を三月十日に確認しました。"),
        ],
        decisions=[
            Decision(
                statement="新機能のリリースは3月10日とする",
                source_quote="新機能のリリースは三月十日を予定しています。",
                speaker="鈴木",
            ),
            Decision(
                # Nobody said this. Fluent, plausible, entirely invented — the
                # exact failure the citation check exists to catch.
                statement=FABRICATED_DECISION,
                source_quote="来月中に全社展開することで合意しました。",
                speaker="田中",
            ),
        ],
        action_items=[
            ActionItem(
                task="KPIダッシュボードを更新する",
                owner="佐藤",
                due="金曜日",
                source_quote="はい、金曜日までに対応します。",
                speaker="佐藤",
            ),
            ActionItem(
                task="次回会議を再来週の火曜日に設定する",
                owner="田中",
                due="再来週の火曜日",
                source_quote="では、次回の会議は再来週の火曜日にしましょう。",
                speaker="田中",
            ),
        ],
    )


def demo_llm(transcript: Transcript | None = None) -> MockLLM:
    """A deterministic LLM returning :func:`canned_minutes`."""
    return MockLLM(default_response=canned_minutes(transcript).model_dump_json())
