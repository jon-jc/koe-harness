"""The pty backend.

Skipped where there is no pty — a minimal Windows install without pywinpty is
a supported configuration, not a broken one, and the panel falls back to pipes
there. The tests that do run spawn a real terminal, because the whole claim of
this backend is about how a real program behaves when it asks whether it has
one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from koe.terminal import (
    PtyBackend,
    ShellBackend,
    TerminalService,
    pty_available,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not pty_available(), reason="no pty on this machine"),
]


@pytest.fixture
async def service() -> AsyncIterator[TerminalService]:
    svc = TerminalService()
    svc.register_backend(ShellBackend())
    svc.register_backend(PtyBackend())
    try:
        yield svc
    finally:
        await svc.dispose()


async def read_until(session: object, needle: str, budget: float = 6.0) -> str:
    """Collect output until `needle` appears or the budget runs out."""
    seen = ""
    deadline = asyncio.get_running_loop().time() + budget
    while needle not in seen and asyncio.get_running_loop().time() < deadline:
        chunk = await session.follow(wait_s=0.5)  # type: ignore[attr-defined]
        seen += chunk
    return seen


# --------------------------------------------------------------------------
# the two backends are different products
# --------------------------------------------------------------------------


async def test_both_backends_coexist(service: TerminalService) -> None:
    """Neither is a fallback for the other; a session picks one by type."""
    pipes = await service.open(owner="me", type="shell")
    pty = await service.open(owner="me", type="pty")

    assert not pipes.raw
    assert pty.raw


async def test_a_pty_session_reports_a_tty(service: TerminalService) -> None:
    """The whole point: programs that check will draw rather than refuse."""
    session = await service.open(owner="me", type="pty", size=(90, 25))
    await asyncio.sleep(1.0)
    session.drain()

    await session.channel.write(b"python -c \"import sys;print('TTY', sys.stdout.isatty())\"\r\n")
    seen = await read_until(session, "TTY True")

    assert "TTY True" in seen


async def test_a_pipe_session_does_not(service: TerminalService) -> None:
    """The counterpart, so the distinction is pinned rather than assumed."""
    session = await service.open(owner="me", type="shell")
    outcome = await service.send(
        session.id, 'python -c "import sys;print(sys.stdout.isatty())"', owner="me"
    )

    assert "False" in outcome.output


async def test_pty_output_keeps_its_escape_sequences(service: TerminalService) -> None:
    """They are the payload here, not noise: an emulator needs them all."""
    session = await service.open(owner="me", type="pty", size=(90, 25))
    seen = await read_until(session, "\x1b", budget=4.0)

    assert "\x1b" in seen


# --------------------------------------------------------------------------
# window size
# --------------------------------------------------------------------------


async def test_a_resize_reaches_the_shell(service: TerminalService) -> None:
    """Without this a full-screen program wraps at the old width and looks corrupt."""
    session = await service.open(owner="me", type="pty", size=(80, 24))
    await asyncio.sleep(1.0)
    session.drain()

    session.channel.resize(133, 41)
    await asyncio.sleep(0.4)
    await session.channel.write(b"echo COLS=$(tput cols) ROWS=$(tput lines)\r\n")
    seen = await read_until(session, "COLS=133")

    assert "COLS=133" in seen
    assert "ROWS=41" in seen


async def test_the_opening_size_is_honoured(service: TerminalService) -> None:
    session = await service.open(owner="me", type="pty", size=(111, 33))
    await asyncio.sleep(1.0)
    session.drain()

    await session.channel.write(b"echo W=$(tput cols)\r\n")
    seen = await read_until(session, "W=111")

    assert "W=111" in seen


# --------------------------------------------------------------------------
# one reader
# --------------------------------------------------------------------------


async def test_follow_returns_buffered_output_without_waiting(
    service: TerminalService,
) -> None:
    """A streaming consumer must not have to race the service's own reader.

    Reading the channel directly from a second place silently steals chunks
    from the first — output still arrives, just not all of it, which is the
    kind of bug that survives a test that only checks the connection works.
    """
    session = await service.open(owner="me", type="pty", size=(90, 25))
    await asyncio.sleep(1.0)

    first = await session.follow(wait_s=2.0)
    assert first  # the banner was buffered by the pump and handed over whole

    # And the buffer is now empty, so nothing is delivered twice.
    assert session.peek() == ""


async def test_follow_gives_up_rather_than_hanging(service: TerminalService) -> None:
    session = await service.open(owner="me", type="pty", size=(90, 25))
    await asyncio.sleep(1.0)
    session.drain()

    assert await session.follow(wait_s=0.3) == ""


# --------------------------------------------------------------------------
# teardown
# --------------------------------------------------------------------------


async def test_closing_a_pty_ends_the_process(service: TerminalService) -> None:
    """A leaked pty outlives the app, and nothing else will reap it."""
    session = await service.open(owner="me", type="pty")
    await service.close(session.id, owner="me")

    assert not session.alive
