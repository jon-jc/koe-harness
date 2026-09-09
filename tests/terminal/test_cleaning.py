"""Cleaning terminal output.

Pure functions over strings, so they live apart from the tests that spawn a
shell: these are the fast ones, and they carry no asyncio mark.
"""

from __future__ import annotations

from koe.terminal.session import PROMPT_SENTINEL, clean_output, strip_ansi

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
