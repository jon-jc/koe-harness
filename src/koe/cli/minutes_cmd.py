"""The ``koe minutes`` command: transcript -> 議事録, with verification shown.

Runs the whole flagship path end to end -- ASR, diarization, fusion, LLM,
guardrails -- and prints the verification result alongside the document, so
what the system rejected is as visible as what it produced.

Without ``--live`` the LLM is a deterministic stand-in whose output contains
**one deliberately fabricated decision**. That is not a shortcut; it is the
demo. The guardrail catching an invented claim is the part worth seeing, and a
real model cannot be relied upon to hallucinate on cue.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import typer
from rich.console import Console

from koe.domain.audio import STANDARD_FORMAT, AudioChunk
from koe.domain.transcript import Transcript, attribute_speakers
from koe.evaluation.corpus import ALL_SCRIPTS
from koe.minutes.generator import MinutesGenerator
from koe.minutes.schema import ActionItem, Decision, Minutes, Topic
from koe.providers.mock import MockASR, MockDiarization, MockLLM


async def _build_transcript(script: Any) -> Transcript:
    seconds = max(u.end for u in script)
    audio = AudioChunk(data=bytes(STANDARD_FORMAT.bytes_for(seconds)))
    transcript = await MockASR(script=script, degradation=0.0).transcribe(audio)
    diarization = await MockDiarization(script=script).diarize(audio)
    return attribute_speakers(transcript, diarization)


def _canned_llm(transcript: Transcript) -> MockLLM:
    """A stand-in whose output includes one fabricated decision, on purpose."""
    participants = sorted({s.speaker for s in transcript.final_segments if s.speaker})
    minutes = Minutes(
        language=transcript.dominant_language(),
        title="第三四半期 売上レビュー",
        participants=participants,
        summary="第三四半期の売上は目標を達成しました。KPIダッシュボードの更新期限とリリース日を確認しました。",
        topics=[Topic(title="売上レビュー", summary="前年比120パーセントで目標を達成しました。")],
        decisions=[
            Decision(
                statement="新機能のリリースは3月10日とする",
                source_quote="新機能のリリースは三月十日を予定しています。",
                speaker="鈴木",
            ),
            Decision(
                # Nobody said this. It is fluent, plausible, and entirely invented
                # -- exactly the failure mode the citation check exists to catch.
                statement="来月中に全社展開を完了する",
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
            )
        ],
    )
    return MockLLM(default_response=minutes.model_dump_json())


def register(app: typer.Typer, console: Console) -> None:
    """Attach the ``minutes`` command to `app`."""

    @app.command()
    def minutes(
        meeting: Annotated[
            str, typer.Option(help="Which bundled meeting to summarize")
        ] = "quarterly-ja",
        live: Annotated[
            bool, typer.Option("--live", help="Use Claude (needs ANTHROPIC_API_KEY)")
        ] = False,
    ) -> None:
        """Generate 議事録 from a meeting, with citation verification."""
        script = ALL_SCRIPTS.get(meeting)
        if script is None:
            console.print(f"[red]unknown meeting {meeting!r}[/red]")
            console.print(f"available: {', '.join(sorted(ALL_SCRIPTS))}")
            raise typer.Exit(code=1)

        transcript = asyncio.run(_build_transcript(script))

        console.rule("[bold]1. transcript (ASR x diarization)")
        for segment in transcript.final_segments:
            console.print(f"[cyan]{segment.speaker}[/cyan]: {segment.text}")

        if live:
            from koe.providers.llm.anthropic import AnthropicLLM

            if not AnthropicLLM.is_configured():
                console.print("[red]ANTHROPIC_API_KEY is not set[/red]")
                raise typer.Exit(code=1)
            llm: Any = AnthropicLLM()
            console.print(f"[dim]using {llm.info}[/dim]")
        else:
            llm = _canned_llm(transcript)

        result = asyncio.run(MinutesGenerator(llm).generate(transcript))

        console.rule("[bold]2. 議事録")
        console.print(result.minutes.render())

        console.rule("[bold]3. verification")
        console.print(result.audit())
        if result.dropped and not live:
            console.print(
                "[yellow]The dropped decision was fabricated deliberately, to show "
                "the citation check rejecting an invented claim.[/yellow]"
            )
