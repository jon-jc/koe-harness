"""The ``koe serve`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console


def register(app: typer.Typer, console: Console) -> None:
    """Attach the ``serve`` command to `app`."""

    @app.command()
    def serve(
        host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
        port: Annotated[int, typer.Option(help="Port")] = 8000,
        reload: Annotated[bool, typer.Option("--reload", help="Reload on change")] = False,
    ) -> None:
        """Run the API and the realtime web client."""
        try:
            import uvicorn
        except ImportError:
            console.print("[red]uvicorn is not installed[/red]")
            console.print("  pip install 'koe-harness[api]'")
            raise typer.Exit(code=1) from None

        bundle = Path(__file__).resolve().parents[3] / "web" / "dist" / "app.js"
        if not bundle.exists():
            # Worth saying out loud: the API works without it, but the demo
            # everyone actually wants to see is the browser client.
            console.print(
                "[yellow]web client not built — the API will run, but / will be "
                "a placeholder.[/yellow]\n  cd web && npm install && npm run build\n"
            )

        console.print(f"[bold]koe[/bold] on http://{host}:{port}")
        console.print(f"[dim]websocket: ws://{host}:{port}/v1/stream[/dim]\n")
        uvicorn.run(
            "koe.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
