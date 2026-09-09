"""Workspace file access, fenced to one directory tree.

The code viewer and the model-facing file tools are the same capability seen
from two sides; :mod:`koe.workspace.service` is the single place the
containment check lives, because two implementations of that check means one
of them is wrong.
"""

from koe.workspace.service import (
    IGNORED_DIRS,
    LANGUAGES,
    MAX_READ_BYTES,
    Entry,
    FileView,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "IGNORED_DIRS",
    "LANGUAGES",
    "MAX_READ_BYTES",
    "Entry",
    "FileView",
    "Workspace",
    "WorkspaceError",
]
