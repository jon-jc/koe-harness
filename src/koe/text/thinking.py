"""Removing reasoning-model scratchpads from text meant for a person.

Reasoning models emit their working inside ``<think>`` tags and their answer
outside them. Anything that shows a model's raw output to a user -- koe's chat
panel, a 議事録 summary, a tool result -- has to strip those first, and the
stripping has to survive output that is still streaming.

Ported from OpenWhispr's ``src/helpers/stripThinking.js`` (MIT), whose ordering
is the part worth keeping. Three passes, and each one exists because a simpler
version leaks:

1. **Innermost closed pairs first, repeatedly.** A single non-greedy
   ``<think>.*?</think>`` over nested blocks matches from the outer open tag to
   the inner close, leaving the outer ``</think>`` behind as visible text.
   Removing innermost pairs until none remain collapses nesting from the inside
   out, which cannot strand a tag.

2. **Then an unterminated trailing block.** While a response is still
   streaming, the opening tag has arrived and the closing one has not. Without
   this the user watches the model think.

3. **Then any orphan closing tag.** A model that emits ``</think>`` without an
   opening one is malformed, but the fix is to drop the tag rather than to show
   it.

``<thinking>`` is accepted alongside ``<think>``: both are in use, and a
harness that routes to several providers will meet both.
"""

from __future__ import annotations

import re

#: Both spellings, so one pattern covers every provider we route to.
_TAG = r"think(?:ing)?"

#: An open tag, the shortest run containing no further open tag, then a close.
#: The inner lookahead is what makes it *innermost* rather than merely lazy.
_INNERMOST_PAIR = re.compile(
    rf"<{_TAG}>(?:(?!<{_TAG}>)[\s\S])*?</{_TAG}>",
    re.IGNORECASE,
)
_UNTERMINATED = re.compile(rf"<{_TAG}>[\s\S]*$", re.IGNORECASE)
_ORPHAN_CLOSE = re.compile(rf"</{_TAG}>", re.IGNORECASE)


def _collapse_pairs(text: str) -> str:
    """Remove closed blocks innermost-first until none are left."""
    out = text
    while True:
        collapsed = _INNERMOST_PAIR.sub("", out)
        if collapsed == out:
            return out
        out = collapsed


def strip_thinking(text: str) -> str:
    """Return `text` with reasoning blocks removed.

    Safe on partial output: a block that has opened and not yet closed is
    removed along with everything after it, so a streaming response shows the
    answer appearing rather than the reasoning that precedes it.
    """
    if not text:
        return text
    out = _UNTERMINATED.sub("", _collapse_pairs(text))
    return _ORPHAN_CLOSE.sub("", out).strip()


def has_open_thinking(text: str) -> bool:
    """Whether `text` ends inside an unterminated reasoning block.

    A streaming UI uses this to say "thinking..." instead of rendering an empty
    bubble, which is otherwise indistinguishable from a stalled request.
    """
    return bool(_UNTERMINATED.search(_collapse_pairs(text or "")))
