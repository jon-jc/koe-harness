"""``koe vad-bench``: score the detector, with a control column."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from koe.evaluation.vad import VADReport
from koe.evaluation.vadbench import run, scenes
from koe.pipeline.vad import VADConfig

#: The table needs this much room to stay readable. Rich wraps to the
#: terminal otherwise, and a benchmark whose numbers are split across two lines
#: each is a benchmark nobody reads.
WIDTH = 104


def _table(console: Console, title: str, report: VADReport) -> None:
    table = Table(title=title, header_style="bold")
    table.add_column("condition", no_wrap=True)
    for column in ("P", "R", "F1", "found", "miss", "FA", "split", "merge", "on ms", "off ms"):
        table.add_column(column, justify="right")

    rows = [*report.scores.items(), ("TOTAL", report.total)]
    for name, score in rows:
        style = "bold" if name == "TOTAL" else None
        table.add_row(
            name,
            f"{score.precision:.3f}",
            f"{score.recall:.3f}",
            f"{score.f1:.3f}",
            f"{score.detected}/{score.reference_segments}",
            str(score.missed),
            str(score.false_alarms),
            str(score.split),
            str(score.merged),
            f"{score.onset_bias_ms:.0f}",
            f"{score.offset_bias_ms:.0f}",
            style=style,
        )
    console.print(table)


def register(app: typer.Typer, console: Console) -> None:
    @app.command("vad-bench")
    def vad_bench(
        control: Annotated[
            bool,
            typer.Option(help="Also score with voicing off, as a baseline."),
        ] = True,
    ) -> None:
        """Score voice activity detection across eight room conditions.

        Reports the control column by default. A benchmark read only as a delta
        hides a broken baseline, which has happened here at least once.
        """
        # A console wide enough for the table, whatever the terminal is.
        out = console if console.width >= WIDTH else Console(width=WIDTH)

        built = scenes()
        audio = sum(scene.duration_s for scene in built.values())
        out.print(f"[dim]{len(built)} conditions, {audio:.0f}s of audio[/dim]")

        after = run(VADConfig(require_voicing=True), built)
        if not control:
            _table(out, "koe", after)
            return

        before = run(VADConfig(require_voicing=False), built)
        _table(out, "energy only (control)", before)
        _table(out, "energy + voicing", after)

        b: Any = before.total
        a: Any = after.total
        console.print(
            f"\n  precision   {b.precision:.3f} → [bold]{a.precision:.3f}[/bold]"
            f"   ({a.precision - b.precision:+.3f})"
            f"\n  recall      {b.recall:.3f} → [bold]{a.recall:.3f}[/bold]"
            f"   ({a.recall - b.recall:+.3f})"
            f"\n  F1          {b.f1:.3f} → [bold]{a.f1:.3f}[/bold]"
            f"   ({a.f1 - b.f1:+.3f})"
            f"\n  false alarms per minute   {b.false_alarms_per_minute:.2f} → "
            f"[bold]{a.false_alarms_per_minute:.2f}[/bold]"
            f"\n  merged utterances         {b.merged} → [bold]{a.merged}[/bold]"
        )
