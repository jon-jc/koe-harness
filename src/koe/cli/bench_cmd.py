"""The ``koe bench`` command."""

from __future__ import annotations

import platform
from typing import Annotated

import typer
from rich.console import Console


def register(app: typer.Typer, console: Console) -> None:
    """Attach the ``bench`` command to `app`."""

    @app.command()
    def bench(
        only: Annotated[
            str | None, typer.Option(help="Substring filter on group or benchmark name")
        ] = None,
        markdown: Annotated[bool, typer.Option("--markdown", help="Emit a Markdown table")] = False,
    ) -> None:
        """Run the performance benchmarks."""
        try:
            from benchmarks.runner import render, render_markdown, summarize
            from benchmarks.suite import bench as suite
        except ImportError:
            console.print(
                "[red]benchmarks are not importable[/red]\n"
                "  run from a source checkout: python -m benchmarks"
            )
            raise typer.Exit(code=1) from None

        from koe.text.tokenize import mecab_available

        console.print(
            f"[dim]Python {platform.python_version()} on {platform.system()} "
            f"{platform.machine()}, MeCab "
            f"{'available' if mecab_available() else 'unavailable'}[/dim]"
        )
        results = suite.run(only=only)
        if markdown:
            print(render_markdown(results))
        else:
            print(render(results))
            print(summarize(results))
