"""Assembling the system prompt from what is actually mounted.

Ported from DeepSeek Harness's `dsh-system-prompt` (MIT). koe's prompt was a
module constant that described tools generically -- "You have tools for reading
this workspace's files, running shell commands..." -- and it said that whether
or not those plugins were loaded.

That is wrong in both directions, and neither failure is loud. Turn the terminal
plugin off and the model is still told it can run commands, so it tries, and
gets a refusal it was given no way to anticipate. Add a plugin and the model is
told nothing about it, so the tool sits in the schema list with no instructions
about when to reach for it.

**A contribution is an effect, like a tool registration.** `section()` returns a
disposer, so the plugin that added the instructions is the only thing that can
remove them, and unloading it takes its prompt text with it. The prompt then
describes the harness that exists rather than the one someone wrote about once.

**Order is a number, and the numbers are allocated centrally.** Identity first,
then policy, then per-tool instructions, then deployment-specific text last.
Ties break on name so the assembly is deterministic -- a prompt whose section
order depends on plugin load order would change its bytes between runs, and a
prompt whose bytes change invalidates a provider's prefix cache for the whole
conversation.

**An unresolved variable fails assembly.** `{{workspace}}` with nothing to put
there is a prompt that says `{{workspace}}` to the model. dsh fails instead of
sending it, and so does this: a malformed prompt is worse than a missing turn,
because it will be silently obeyed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Where each kind of contribution sits. Centrally allocated, as in dsh, so two
#: plugins written by different people do not have to negotiate an order --
#: they name a slot and the numbers do the rest. Gaps are deliberate: a section
#: can be inserted between two existing ones without renumbering either.
ORDERS: dict[str, int] = {
    # Who the assistant is. Always first.
    "IDENTITY": -1000,
    # How it should behave, before it is told what it can do.
    "POLICY": 500,
    "LANGUAGE": 600,
    # One slot per tool group, in the order a reader would want them.
    "TOOL_WORKSPACE": 1000,
    "TOOL_WRITE": 1100,
    "TOOL_TERMINAL": 1200,
    "TOOL_MEETING": 1300,
    "TOOL_VOCABULARY": 1400,
    # Facts about this deployment, after the reusable instructions.
    "WORKSPACE": 9000,
    "PERSONA": 10_000,
}

#: `{{name}}`, the same spelling dsh uses.
_VARIABLE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


@dataclass(frozen=True, slots=True)
class Section:
    """One contribution to the prompt."""

    name: str
    order: int
    #: Text, or a callable evaluated at each assembly. A callable is what lets
    #: a section describe the state of the thing it belongs to -- how many
    #: terminals are open, which workspace is mounted -- rather than describing
    #: it as it was at registration.
    text: str | Callable[[], str]
    #: Treat this as the entire prompt. More than one is an error: two
    #: contributions each claiming to be the whole thing cannot both be right,
    #: and picking one silently would make the prompt depend on load order.
    complete: bool = False

    def resolve(self) -> str:
        if callable(self.text):
            try:
                return str(self.text() or "")
            except Exception:
                # A section that cannot describe itself must not take the turn
                # with it. Its absence is a worse prompt; its exception is no
                # prompt at all.
                logger.exception("prompt section %r failed to render", self.name)
                return ""
        return str(self.text or "")


class PromptAssemblyError(Exception):
    """Assembly refused to produce a prompt."""


@dataclass(slots=True)
class SystemPrompt:
    """The prompt registry: sections, variables, and the assembly."""

    _sections: dict[str, Section] = field(default_factory=dict, init=False, repr=False)
    _variables: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # -- registration -----------------------------------------------------

    def section(
        self,
        name: str,
        text: str | Callable[[], str],
        *,
        order: int | str = "POLICY",
        complete: bool = False,
    ) -> Callable[[], None]:
        """Contribute a section. Returns its disposer.

        A disposer rather than `remove(name)` for the reason the tool registry
        gives: the plugin that added it holds the only handle that removes it,
        so unloading a plugin cannot leave text behind describing capabilities
        that left with it.

        `order` accepts a slot name from :data:`ORDERS` or a raw number.
        """
        if name in self._sections:
            raise ValueError(f"prompt section {name!r} is already registered")

        resolved = ORDERS.get(order, 0) if isinstance(order, str) else int(order)
        entry = Section(name=name, order=resolved, text=text, complete=complete)
        self._sections[name] = entry

        def dispose() -> None:
            # Bound to this exact object, so a stale disposer cannot remove a
            # later section that reused the name.
            if self._sections.get(name) is entry:
                del self._sections[name]

        return dispose

    def variable(self, name: str, value: str) -> None:
        """Set a value for `{{name}}`."""
        self._variables[name] = value

    def variables(self, **values: str) -> None:
        for name, value in values.items():
            self.variable(name, value)

    # -- assembly ---------------------------------------------------------

    def sections(self) -> list[Section]:
        """Every section, in assembly order.

        Ordered by slot, then by name. The name tie-break is what makes the
        output deterministic: without it two sections sharing a slot would
        assemble in dictionary order, the prompt's bytes would depend on plugin
        load order, and a provider's prefix cache would miss on a run that
        changed nothing.
        """
        return sorted(self._sections.values(), key=lambda entry: (entry.order, entry.name))

    def render(self, **overrides: str) -> str:
        """The assembled prompt.

        Raises :class:`PromptAssemblyError` when a section claims to be the
        whole prompt and is not alone, or when a variable has no value. Both
        are refusals to send something malformed: a prompt containing a literal
        `{{workspace}}` is not a degraded prompt, it is an instruction the model
        will try to make sense of.
        """
        ordered = self.sections()
        complete = [entry for entry in ordered if entry.complete]
        if len(complete) > 1:
            names = ", ".join(entry.name for entry in complete)
            raise PromptAssemblyError(
                f"more than one section claims to be the complete prompt: {names}"
            )
        if complete:
            ordered = complete

        parts = [text for entry in ordered if (text := entry.resolve().strip())]
        rendered = "\n\n".join(parts)

        values = {**self._variables, **overrides}
        missing: list[str] = []

        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                missing.append(key)
                return match.group(0)
            return values[key]

        rendered = _VARIABLE.sub(substitute, rendered)
        if missing:
            raise PromptAssemblyError(
                f"prompt has unresolved variables: {', '.join(sorted(set(missing)))}"
            )
        return rendered

    def describe(self) -> list[dict[str, Any]]:
        """Every section, for the settings panel.

        Worth exposing: "why is the assistant behaving like that" is usually
        answered by the prompt, and a prompt assembled from a dozen plugins is
        otherwise invisible.
        """
        return [
            {
                "name": entry.name,
                "order": entry.order,
                "complete": entry.complete,
                "dynamic": callable(entry.text),
                "characters": len(entry.resolve()),
            }
            for entry in self.sections()
        ]

    def __len__(self) -> int:
        return len(self._sections)
