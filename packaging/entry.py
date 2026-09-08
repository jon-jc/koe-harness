"""Frozen-application entry point.

A thin module rather than pointing PyInstaller at ``koe/desktop/app.py``
directly: the analyser treats the entry script as a script rather than a
package member, and importing through the package keeps relative imports and
``__package__`` behaving the way they do in a source run.
"""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    # Required before anything may spawn a process in a frozen build; without
    # it a child re-executes the bootloader and forks the app repeatedly.
    multiprocessing.freeze_support()

    from koe.desktop.app import main

    sys.exit(main())
