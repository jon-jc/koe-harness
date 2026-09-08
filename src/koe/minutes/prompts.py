"""Prompts for 議事録 generation.

Written per language rather than translated, because the register differs. A
Japanese 議事録 is a semi-formal document with established conventions --
です・ます体, 体言止め for headings, decisions stated flatly without hedging --
and a prompt translated from English produces text that is grammatically fine
and reads wrong to a Japanese reader.

Two instructions do most of the quality work:

* **Cite everything.** Every decision and action item must carry the verbatim
  span it came from. This is enforced downstream, but stating it here also
  reduces invention at the source -- there is nowhere to put an embellishment
  when a quote has to back it.
* **Leave it empty rather than guess.** Models are strongly inclined to fill an
  `owner` or `due` field because the schema has one. An invented deadline is
  worse than a missing one, so the prompt says so explicitly, repeatedly, and
  the guardrails verify it.
"""

from __future__ import annotations

from koe.text.script import Language

_SHARED_RULES_JA = """
## 厳守すべきルール

1. **引用の義務**: 決定事項 (decisions) とアクションアイテム (action_items) には、
   必ず `source_quote` に文字起こしからの**逐語引用**を入れてください。
   言い換え・翻訳・整形をしてはいけません。原文をそのままコピーしてください。
2. **推測の禁止**: 発言されていないことを書かないでください。
   - 担当者が明示されていない場合、`owner` は空文字列にしてください。
   - 期限が述べられていない場合、`due` は空文字列にしてください。
   - 「たぶんこうだろう」で埋めることは、空欄より有害です。
3. **話者名**: `owner` と `speaker` には、文字起こしに現れる話者ラベルを
   そのまま使ってください。存在しない人物を作らないでください。
4. **文体**: です・ます体で記述してください。議事録として自然な日本語にしてください。
5. **英語混在**: 発言に英語が含まれる場合、専門用語はそのまま残してください。
   無理に翻訳しないでください。
""".strip()

_SHARED_RULES_EN = """
## Rules you must follow

1. **Cite everything**: every entry in `decisions` and `action_items` must carry a
   `source_quote` copied **verbatim** from the transcript. Do not paraphrase,
   translate, or tidy it up. Copy the exact characters.
2. **Never infer**: do not write anything that was not said.
   - If no one explicitly took an action on, leave `owner` empty.
   - If no deadline was stated, leave `due` empty.
   - A guessed value is worse than a missing one.
3. **Speaker names**: use the speaker labels exactly as they appear in the
   transcript. Do not invent participants.
4. **Register**: write plainly and factually, as minutes rather than prose.
""".strip()

SYSTEM_JA = f"""
あなたは日本企業の会議の議事録を作成する専門家です。
文字起こし (音声認識の結果) を読み、構造化された議事録を JSON で出力してください。

音声認識の出力には誤認識が含まれる場合があります。明らかな誤変換は文脈から
補って解釈して構いませんが、**存在しない内容を追加してはいけません**。

{_SHARED_RULES_JA}
""".strip()

SYSTEM_EN = f"""
You write minutes for business meetings. Read the transcript (produced by speech
recognition) and return structured minutes as JSON.

Speech recognition output contains errors. You may interpret an obvious
misrecognition from context, but you must **never add content that is not there**.

{_SHARED_RULES_EN}
""".strip()


def system_prompt(language: Language) -> str:
    """The system prompt for a meeting in `language`.

    Returned as a stable string with no per-request content, which is what
    makes it worth caching: it is byte-identical across every meeting, so after
    the first call it bills at a fraction of the input rate.
    """
    return SYSTEM_JA if language in (Language.JA, Language.MIXED) else SYSTEM_EN


def user_prompt(transcript: str, language: Language, *, participants: list[str]) -> str:
    """The per-meeting prompt, which is the only part that varies."""
    roster = "、".join(participants) if language is Language.JA else ", ".join(participants)
    if language in (Language.JA, Language.MIXED):
        header = "以下は会議の文字起こしです。"
        roster_line = f"出席者: {roster}" if participants else ""
        instruction = "この文字起こしから議事録を作成してください。"
    else:
        header = "Here is the transcript of a meeting."
        roster_line = f"Participants: {roster}" if participants else ""
        instruction = "Produce the minutes for this meeting."

    parts = [header]
    if roster_line:
        parts.append(roster_line)
    parts += ["", "<transcript>", transcript, "</transcript>", "", instruction]
    return "\n".join(parts)


def repair_prompt(problems: list[str], language: Language) -> str:
    """Follow-up asking the model to fix specific unsupported claims.

    Names the offending claims rather than asking for a general retry, because
    "try again" on a non-deterministic model is a coin flip, while "these three
    quotes are not in the transcript" is a correctable instruction.
    """
    listing = "\n".join(f"- {p}" for p in problems)
    if language in (Language.JA, Language.MIXED):
        return (
            "以下の項目は、文字起こしに存在しない内容を含んでいます。\n\n"
            f"{listing}\n\n"
            "これらを削除するか、文字起こしからの正確な逐語引用に修正して、"
            "議事録全体を再度出力してください。"
        )
    return (
        "The following entries contain content that is not in the transcript:\n\n"
        f"{listing}\n\n"
        "Remove them, or correct them with an exact verbatim quote from the "
        "transcript, and return the complete minutes again."
    )
