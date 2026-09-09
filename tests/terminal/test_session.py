"""Terminal sessions.

The tests that matter are the ownership fence, the single-send rule, and the
settlement reason — the three places where getting it wrong is not a bug you
notice, it is a bug that quietly does the wrong thing.

Every test here spawns a real shell. That is deliberate: the interesting
behaviour of this module is its interaction with a process's buffering and
prompt, and a mocked process would only test the mock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from koe.terminal import (
    ShellBackend,
    TerminalError,
    TerminalFailure,
    TerminalService,
    WaitReason,
)
from koe.terminal.session import PROMPT_SENTINEL, clean_output, strip_ansi

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def service() -> AsyncIterator[TerminalService]:
    svc = TerminalService()
    svc.register_backend(ShellBackend())
    try:
        yield svc
    finally:
        await svc.dispose()


# --------------------------------------------------------------------------
# cleaning -- pure, so it does not need a shell
# --------------------------------------------------------------------------


def test_colour_codes_are_removed() -> None:
    assert strip_ansi("\x1b[32mgreen\x1b[0m") == "green"


def test_window_title_sequences_are_removed() -> None:
    """OSC ends at a BEL, not at a letter, so one regex cannot catch both."""
    assert strip_ansi("\x1b]0;some title\x07text") == "text"


def test_carriage_returns_are_folded() -> None:
    """Progress output rewriting one line reads as one line once escapes go."""
    assert strip_ansi("a\r\nb\rc") == "a\nbc"


def test_the_echoed_command_is_dropped() -> None:
    """The caller typed it; returning it costs tokens and reads as noise."""
    raw = f"ls -la\ntotal 0\n{PROMPT_SENTINEL}\n"
    assert clean_output(raw, command="ls -la") == "total 0"


def test_a_line_that_merely_looks_like_the_command_survives() -> None:
    """Only the first line is the echo; a later match is real output."""
    raw = f"echo hi\nhi\necho hi\n{PROMPT_SENTINEL}\n"
    assert clean_output(raw, command="echo hi") == "hi\necho hi"


def test_the_prompt_never_reaches_the_caller() -> None:
    assert PROMPT_SENTINEL not in clean_output(f"out\n{PROMPT_SENTINEL}\n")


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


async def test_a_session_opens_and_runs_a_command(service: TerminalService) -> None:
    session = await service.open(owner="me")
    outcome = await service.send(session.id, "echo hello-from-koe", owner="me")

    assert outcome.output == "hello-from-koe"
    assert outcome.reason is WaitReason.PROMPT


async def test_settlement_is_proof_rather_than_a_guess(service: TerminalService) -> None:
    """The prompt means the command returned; silence only means silence."""
    session = await service.open(owner="me")
    outcome = await service.send(session.id, "echo one", owner="me")

    assert outcome.reason is WaitReason.PROMPT
    # And it is fast, because it did not have to sit out an idle window.
    assert outcome.duration_ms < 2000


async def test_state_persists_between_sends(service: TerminalService) -> None:
    """The entire reason a session exists rather than a one-shot exec."""
    session = await service.open(owner="me")
    await service.send(session.id, "cd src", owner="me")
    outcome = await service.send(session.id, "pwd", owner="me")

    assert outcome.output.endswith("/src")


async def test_an_environment_variable_survives_too(service: TerminalService) -> None:
    session = await service.open(owner="me")
    await service.send(session.id, "export KOE_TEST=stateful", owner="me")
    outcome = await service.send(session.id, "echo $KOE_TEST", owner="me")

    assert outcome.output == "stateful"


async def test_stderr_is_captured_with_stdout(service: TerminalService) -> None:
    """A terminal that loses error output is worse than useless."""
    session = await service.open(owner="me")
    outcome = await service.send(session.id, "echo oops >&2", owner="me")

    assert "oops" in outcome.output


# --------------------------------------------------------------------------
# ownership
# --------------------------------------------------------------------------


async def test_another_owner_cannot_touch_the_session(service: TerminalService) -> None:
    """Learning an id from a log is not authorization."""
    session = await service.open(owner="alice")

    with pytest.raises(TerminalFailure) as caught:
        await service.send(session.id, "whoami", owner="mallory")

    assert caught.value.code is TerminalError.FOREIGN_SESSION


async def test_an_unknown_session_is_distinguished_from_a_foreign_one(
    service: TerminalService,
) -> None:
    """The UI shows a different thing for a stale id than for someone else's."""
    with pytest.raises(TerminalFailure) as caught:
        service.read("t_nonexistent", owner="me")

    assert caught.value.code is TerminalError.NO_SESSION


async def test_listing_is_scoped_to_the_owner(service: TerminalService) -> None:
    await service.open(owner="alice")
    await service.open(owner="bob")

    assert len(service.list(owner="alice")) == 1
    assert len(service.list()) == 2


# --------------------------------------------------------------------------
# concurrency and failure
# --------------------------------------------------------------------------


async def test_a_second_send_is_refused_while_one_is_settling(
    service: TerminalService,
) -> None:
    """Two writers on one stdin produce a line neither of them meant to type."""
    session = await service.open(owner="me")
    slow = asyncio.create_task(service.send(session.id, "sleep 1", owner="me", timeout_s=3))
    await asyncio.sleep(0.2)

    with pytest.raises(TerminalFailure) as caught:
        await service.send(session.id, "echo second", owner="me")

    assert caught.value.code is TerminalError.SEND_ACTIVE
    await slow


async def test_a_send_that_outlasts_its_budget_reports_a_timeout(
    service: TerminalService,
) -> None:
    """Not an error: the command is still running, and the caller can read on."""
    session = await service.open(owner="me")
    outcome = await service.send(session.id, "sleep 3", owner="me", timeout_s=0.4)

    assert outcome.reason is WaitReason.TIMEOUT


async def test_reading_after_a_timeout_follows_the_command(
    service: TerminalService,
) -> None:
    session = await service.open(owner="me")
    await service.send(session.id, "sleep 0.5; echo finished", owner="me", timeout_s=0.2)
    await asyncio.sleep(1.2)

    assert "finished" in service.read(session.id, owner="me")


async def test_an_exited_shell_is_reported_rather_than_hung(
    service: TerminalService,
) -> None:
    session = await service.open(owner="me")
    await service.send(session.id, "exit", owner="me", timeout_s=3)
    await asyncio.sleep(0.3)

    with pytest.raises(TerminalFailure) as caught:
        await service.send(session.id, "echo after", owner="me")

    assert caught.value.code is TerminalError.SESSION_EXITED


async def test_an_unknown_backend_names_the_alternatives(service: TerminalService) -> None:
    with pytest.raises(TerminalFailure) as caught:
        await service.open(type="containers")

    assert caught.value.code is TerminalError.NO_BACKEND
    assert "shell" in str(caught.value)


async def test_the_session_cap_refuses_rather_than_evicts() -> None:
    """The caller knows which of their sessions is finished; we do not."""
    svc = TerminalService(max_sessions=1)
    svc.register_backend(ShellBackend())
    try:
        await svc.open(owner="me")
        with pytest.raises(TerminalFailure) as caught:
            await svc.open(owner="me")
        assert caught.value.code is TerminalError.TOO_MANY
    finally:
        await svc.dispose()


# --------------------------------------------------------------------------
# teardown
# --------------------------------------------------------------------------


async def test_closing_ends_the_process(service: TerminalService) -> None:
    session = await service.open(owner="me")
    await service.close(session.id, owner="me")

    assert not session.alive


async def test_dispose_closes_everything(service: TerminalService) -> None:
    """A leaked shell outlives the app that spawned it."""
    first = await service.open(owner="me")
    second = await service.open(owner="me")

    await service.dispose()

    assert not first.alive
    assert not second.alive
    assert service.list() == []


async def test_reap_forgets_dead_sessions(service: TerminalService) -> None:
    session = await service.open(owner="me")
    await service.close(session.id, owner="me")

    assert service.reap() == 1
    assert service.list() == []


async def test_a_runaway_command_does_not_grow_without_bound(
    service: TerminalService,
) -> None:
    """The buffer keeps the most recent output and reports what it dropped."""
    session = await service.open(owner="me")
    await service.send(
        session.id,
        "for i in $(seq 1 40000); do echo 'line of output padding padding padding'; done",
        owner="me",
        timeout_s=25,
    )

    assert session.dropped > 0
    assert session._buffered <= 300_000
