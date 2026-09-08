"""The ``koe`` command line.

Evaluation is only useful if running it is easier than not running it. These
commands work on a clean checkout with no credentials and no audio, so there is
no setup step between "I changed a prompt" and "I know whether it got worse".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from koe.evaluation.corpus import build_corpus
from koe.evaluation.dataset import Dataset
from koe.evaluation.regression import DEFAULT_GATES, Gate, Metric, check_regression
from koe.evaluation.runner import EvalReport, mock_transcriber, run_evaluation
from koe.evaluation.statistics import paired_bootstrap
from koe.providers.mock import MockASR
from koe.text.tokenize import mecab_available

app = typer.Typer(
    name="koe",
    help="声 — a bilingual (JA/EN) voice-AI harness.",
    no_args_is_help=True,
    add_completion=False,
)
# The corpus is Japanese, and Windows terminals still default to a legacy
# codepage. Reconfiguring here means `koe evaluate` prints 議事録 rather than
# dying in the encoder halfway through a report.
with contextlib.suppress(AttributeError, OSError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

console = Console()

DEFAULT_DATASET = Path("datasets/meetings.jsonl")


def _bundled_dataset() -> Dataset:
    """The in-repo sample corpus, used when no dataset file is given."""
    return build_corpus()


def _load(path: Path | None) -> Dataset:
    if path is None:
        return (
            _bundled_dataset()
            if not DEFAULT_DATASET.exists()
            else Dataset.from_jsonl(DEFAULT_DATASET)
        )
    return Dataset.from_jsonl(path)


def _run(dataset: Dataset, *, degradation: float, name: str, seed: int) -> EvalReport:
    asr = MockASR(degradation=degradation, name=name, cost_per_audio_minute_usd=0.006)
    return asyncio.run(run_evaluation(dataset, mock_transcriber(asr), system=name, seed=seed))


def _render_report(report: EvalReport, *, seed: int) -> None:
    overall = report.overall(seed=seed)

    header = Table.grid(padding=(0, 2))
    header.add_row("[bold]system[/bold]", report.system)
    header.add_row("[bold]dataset[/bold]", f"{report.dataset} ({len(report.results)} cases)")
    header.add_row("[bold]tokenizer[/bold]", overall.wer.tokenizer or "-")
    header.add_row(
        "[bold]failures[/bold]",
        f"{len(report.failures)} ({report.failure_rate:.1%})",
    )
    console.print(header)
    console.print()

    table = Table(title="quality by slice", header_style="bold")
    table.add_column("slice")
    table.add_column("n", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("DER", justify="right")

    rows = [overall, *report.slice_by_language(seed=seed), *report.slice_by_tag(seed=seed)]
    for row in rows:
        style = "bold" if row.name == "overall" else ""
        table.add_row(
            row.name,
            str(row.n),
            f"{row.cer.interval.point:.2%}",
            f"[{row.cer.interval.lower:.2%}, {row.cer.interval.upper:.2%}]",
            f"{row.wer.interval.point:.2%}",
            f"{row.der:.2%}" if row.der is not None else "-",
            style=style,
        )
    console.print(table)

    ops = Table(title="cost & latency", header_style="bold")
    ops.add_column("metric")
    ops.add_column("value", justify="right")
    ops.add_row("audio processed", f"{report.total_audio_seconds:.0f}s")
    ops.add_row("cost / audio hour", f"${report.cost_per_audio_hour:.4f}")
    ops.add_row("latency p50", f"{report.latency_percentile(0.50):.0f}ms")
    ops.add_row("latency p95", f"{report.latency_percentile(0.95):.0f}ms")
    ops.add_row("mean RTF", f"{report.mean_rtf:.3f}")
    console.print(ops)


@app.command()
def info() -> None:
    """Show which optional backends are available in this environment."""
    table = Table(header_style="bold")
    table.add_column("component")
    table.add_column("status")
    table.add_row(
        "Japanese tokenizer",
        "[green]MeCab (unidic)[/green]"
        if mecab_available()
        else "[yellow]character fallback[/yellow]",
    )
    table.add_row("primary JA metric", "CER")
    table.add_row("primary EN metric", "WER")
    console.print(table)
    if not mecab_available():
        console.print(
            "\n[dim]Install the Japanese extra for morphological segmentation:[/dim]\n"
            "  pip install 'koe-harness[ja]'"
        )


@app.command()
def evaluate(
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d", help="JSONL dataset")] = None,
    degradation: Annotated[float, typer.Option(help="Simulated ASR error rate")] = 0.08,
    system: Annotated[str, typer.Option(help="Label for this run")] = "candidate",
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    save: Annotated[Path | None, typer.Option(help="Write the report as JSON")] = None,
) -> None:
    """Score a system against a dataset, sliced by language and tag."""
    data = _load(dataset)
    report = _run(data, degradation=degradation, name=system, seed=seed)
    _render_report(report, seed=seed)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"\n[dim]wrote {save}[/dim]")


@app.command()
def compare(
    baseline: Annotated[float, typer.Option(help="Baseline simulated error rate")] = 0.08,
    candidate: Annotated[float, typer.Option(help="Candidate simulated error rate")] = 0.10,
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d")] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """A/B two systems on the same data, with a significance verdict."""
    data = _load(dataset)
    base_report = _run(data, degradation=baseline, name="baseline", seed=seed)
    cand_report = _run(data, degradation=candidate, name="candidate", seed=seed)

    from koe.evaluation.regression import paired_samples

    base_samples, cand_samples = paired_samples(base_report, cand_report, metric=Metric.CER)
    comparison = paired_bootstrap(base_samples, cand_samples, seed=seed)

    table = Table(title="paired comparison (CER)", header_style="bold")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("baseline", f"{comparison.baseline:.2%}")
    table.add_row("candidate", f"{comparison.candidate:.2%}")
    table.add_row("delta", f"{comparison.delta:+.2%}")
    table.add_row(
        "95% CI on delta",
        f"[{comparison.delta_interval.lower:+.2%}, {comparison.delta_interval.upper:+.2%}]",
    )
    table.add_row("p-value", f"{comparison.p_value:.4f}")
    table.add_row("paired cases", str(comparison.n))
    console.print(table)

    colour = {"improvement": "green", "regression": "red", "inconclusive": "yellow"}[
        comparison.verdict
    ]
    console.print(f"\n[{colour}][bold]{comparison.verdict.upper()}[/bold][/{colour}]")
    if comparison.verdict == "inconclusive":
        console.print(
            "[dim]The difference is not distinguishable from run-to-run noise "
            "on this many cases.[/dim]"
        )


@app.command()
def gate(
    baseline: Annotated[float, typer.Option(help="Baseline simulated error rate")] = 0.08,
    candidate: Annotated[float, typer.Option(help="Candidate simulated error rate")] = 0.10,
    max_cer: Annotated[float | None, typer.Option(help="Absolute CER ceiling")] = None,
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d")] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Run the regression gates; exits non-zero on failure, for CI."""
    data = _load(dataset)
    base_report = _run(data, degradation=baseline, name="baseline", seed=seed)
    cand_report = _run(data, degradation=candidate, name="candidate", seed=seed)

    gates = list(DEFAULT_GATES)
    if max_cer is not None:
        gates.append(Gate(metric=Metric.CER, max_value=max_cer))

    result = check_regression(cand_report, base_report, gates, seed=seed)

    for entry in result.results:
        colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[entry.status]
        console.print(
            f"[{colour}]{entry.status:<4}[/{colour}] {entry.gate.name:<24} {entry.detail}"
        )

    if result.passed:
        console.print("\n[green][bold]gates passed[/bold][/green]")
    else:
        console.print("\n[red][bold]gates failed[/bold][/red]")
        raise typer.Exit(code=1)


@app.command("build-dataset")
def build_dataset(
    out: Annotated[Path, typer.Option(help="Where to write the JSONL")] = DEFAULT_DATASET,
) -> None:
    """Write the bundled sample corpus to disk."""
    data = _bundled_dataset()
    data.to_jsonl(out)
    console.print(f"wrote [bold]{len(data)}[/bold] cases to {out}")
    console.print(f"tags: {', '.join(data.tags)}")


@app.command()
def demo(seed: Annotated[int, typer.Option()] = 0) -> None:
    """Show the harness end to end: a good backend, a cheap one, and the verdict."""
    data = _bundled_dataset()

    console.rule("[bold]1. a clean backend")
    _render_report(_run(data, degradation=0.0, name="premium", seed=seed), seed=seed)

    console.rule("[bold]2. a cheaper, noisier backend")
    _render_report(_run(data, degradation=0.12, name="budget", seed=seed), seed=seed)

    console.rule("[bold]3. is the difference real?")
    compare(baseline=0.0, candidate=0.12, dataset=None, seed=seed)


from koe.cli import minutes_cmd  # noqa: E402

minutes_cmd.register(app, console)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
