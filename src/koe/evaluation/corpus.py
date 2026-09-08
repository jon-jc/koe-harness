"""The bundled evaluation corpus.

Two decisions here, both driven by what the statistics need.

**Cases are utterances, not meetings.** The bootstrap resamples cases, so the
number of cases *is* the sample size. A corpus of three meetings gives n=3, and
a paired permutation test on n=3 has only 2³ = 8 possible sign flips -- a
two-sided p-value cannot go below ~0.25 no matter how large the true effect is.
Significance is unreachable by construction. Scoring per utterance gives enough
cases for the tests to be able to resolve anything at all, and it matches the
resampling unit the statistics layer assumes.

**Tags are derived from content, not hand-assigned.** A tag that has to be
maintained by hand rots. Deriving them means every utterance added to the
corpus is automatically classified, and the slice breakdown stays honest as the
corpus grows.

The material is written to contain what actually breaks Japanese voice
pipelines: numerals in all three written forms, clause-level code-switching,
keigo, dates, and the action-item phrasing a 議事録 feature has to get right.
"""

from __future__ import annotations

from collections.abc import Sequence

from koe.domain.transcript import SpeakerTurn
from koe.evaluation.dataset import Dataset, EvalCase, ScriptLine
from koe.providers.mock import MEETING_EN, MEETING_JA, MEETING_MIXED, ScriptedUtterance
from koe.text.numbers import contains_numeral
from koe.text.script import Language, code_switch_ratio, primary_language

# --------------------------------------------------------------------------
# additional material
# --------------------------------------------------------------------------


def _lines(rows: Sequence[tuple[str, str]], *, start: float = 0.0) -> list[ScriptedUtterance]:
    out: list[ScriptedUtterance] = []
    cursor = start
    for speaker, text in rows:
        out.append(ScriptedUtterance(speaker=speaker, text=text, start=cursor, end=cursor + 3.8))
        cursor += 4.0
    return out


#: An incident review. Dense with numbers, times and system names -- the
#: vocabulary an ASR model trained on general speech handles worst.
INCIDENT_JA: list[ScriptedUtterance] = _lines(
    [
        ("山本", "昨日の障害について報告します。"),
        ("山本", "発生時刻は十四時三十五分、復旧は十六時二十分でした。"),
        ("中村", "影響を受けたユーザーは約三千二百人です。"),
        ("山本", "原因はデータベースの接続プールの枯渇でした。"),
        ("中村", "監視アラートの閾値を見直す必要があります。"),
        ("山本", "再発防止策は今週中にまとめて共有します。"),
        ("中村", "承知しました。レビューをお願いします。"),
    ]
)

#: Planning, with dates and commitments -- what the minutes feature extracts.
PLANNING_JA: list[ScriptedUtterance] = _lines(
    [
        ("佐藤", "来期の開発計画について議論しましょう。"),
        ("田中", "第一四半期はインフラの刷新に注力します。"),
        ("佐藤", "予算は五千万円を想定しています。"),
        ("田中", "採用は四月から三名を予定しています。"),
        ("佐藤", "詳細な計画書は二月二十八日までに提出してください。"),
        ("田中", "はい、承知しました。"),
    ]
)

#: Heavier code-switching, the register of a Japanese engineering team.
TECHNICAL_MIXED: list[ScriptedUtterance] = _lines(
    [
        ("田中", "このAPIのlatencyがSLOを超えています。"),
        ("Bob", "The p99 is 340 milliseconds. 目標は200ミリ秒です。"),
        ("田中", "cacheのhit rateを改善すれば下がると思います。"),
        ("佐藤", "来週までにbenchmarkを取ってreviewしましょう。"),
        ("Bob", "Sounds good. I'll prepare the dashboard."),
        ("田中", "では、そのdeadlineでお願いします。"),
    ]
)

#: English, for matched-content comparison.
STANDUP_EN: list[ScriptedUtterance] = _lines(
    [
        ("Alice", "Let's do a quick round of updates."),
        ("Bob", "I finished the migration script yesterday."),
        ("Carol", "I'm blocked on the staging environment."),
        ("Alice", "I'll escalate that this afternoon."),
        ("Bob", "The deploy is scheduled for March 14th."),
        ("Carol", "We processed about 4,500 records in the test run."),
    ]
)

ALL_SCRIPTS: dict[str, list[ScriptedUtterance]] = {
    "quarterly-ja": list(MEETING_JA),
    "quarterly-en": list(MEETING_EN),
    "quarterly-mixed": list(MEETING_MIXED),
    "incident-ja": INCIDENT_JA,
    "planning-ja": PLANNING_JA,
    "technical-mixed": TECHNICAL_MIXED,
    "standup-en": STANDUP_EN,
}


# --------------------------------------------------------------------------
# tagging
# --------------------------------------------------------------------------

_KEIGO_MARKERS = ("ます", "です", "ございます", "いたします", "ください", "承知")
_ACTION_MARKERS = ("までに", "対応します", "お願いします", "提出", "共有します", "予定")
_EN_ACTION_MARKERS = ("i'll", "we'll", "scheduled", "by ", "let's")


def derive_tags(text: str) -> list[str]:
    """Classify an utterance by the properties that matter for slicing."""
    tags: list[str] = []
    language = primary_language(text)
    tags.append("ja" if language is Language.JA else "en")

    if code_switch_ratio(text) >= 0.15:
        tags.append("code-switch")
    else:
        tags.append("monolingual")

    if contains_numeral(text) or any(ch.isdigit() for ch in text):
        tags.append("numerals")

    if language is Language.JA and any(marker in text for marker in _KEIGO_MARKERS):
        tags.append("keigo")

    lowered = text.lower()
    if any(marker in text for marker in _ACTION_MARKERS) or any(
        marker in lowered for marker in _EN_ACTION_MARKERS
    ):
        tags.append("action-item")

    if text.rstrip().endswith(("か。", "?", "？")):
        tags.append("question")

    return tags


def build_corpus(name: str = "meetings") -> Dataset:
    """Build the bundled corpus: one case per utterance, tags derived."""
    cases: list[EvalCase] = []
    for meeting_id, utterances in ALL_SCRIPTS.items():
        for index, utterance in enumerate(utterances, start=1):
            cases.append(
                EvalCase(
                    id=f"{meeting_id}-{index:02d}",
                    script=[
                        ScriptLine(
                            speaker=utterance.speaker,
                            text=utterance.text,
                            start=0.0,
                            end=utterance.end - utterance.start,
                        )
                    ],
                    speakers=[
                        SpeakerTurn(
                            speaker=utterance.speaker,
                            start=0.0,
                            end=utterance.end - utterance.start,
                        )
                    ],
                    tags=derive_tags(utterance.text),
                    notes=meeting_id,
                )
            )
    return Dataset(
        name=name,
        description=(
            "Bilingual JA/EN meeting utterances covering numerals, code-switching, "
            "keigo, dates and action items."
        ),
        cases=cases,
    )


def build_meeting_corpus(name: str = "meetings-full") -> Dataset:
    """Whole meetings as single cases, for diarization scoring.

    Kept separate from the utterance corpus: DER needs multi-speaker spans,
    while ASR error rates need many independent cases, and mixing the two
    granularities in one dataset would make both sets of numbers harder to read.
    """
    cases: list[EvalCase] = []
    for meeting_id, utterances in ALL_SCRIPTS.items():
        cases.append(
            EvalCase(
                id=meeting_id,
                script=[
                    ScriptLine(speaker=u.speaker, text=u.text, start=u.start, end=u.end)
                    for u in utterances
                ],
                speakers=[
                    SpeakerTurn(speaker=u.speaker, start=u.start, end=u.end) for u in utterances
                ],
                tags=sorted({t for u in utterances for t in derive_tags(u.text)}),
            )
        )
    return Dataset(name=name, description="Whole meetings, for diarization.", cases=cases)
