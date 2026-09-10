"""Child processes die with the app, however the app dies.

Tested for real, on Windows only. The property is that a process killed
outright takes its children with it — which is precisely what a unit test
cannot fake, and precisely what left ten orphaned console hosts behind before
this existed.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from koe.desktop import jobs
from koe.desktop.instance import _process_alive

windows = pytest.mark.skipif(sys.platform != "win32", reason="job objects are a Windows mechanism")

PARENT = """
import subprocess, sys
from koe.desktop.jobs import bind_children_to_this_process

if not bind_children_to_this_process():
    print("unbound", flush=True)
    sys.exit(0)
try:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags={flags},
    )
except OSError:
    print("refused", flush=True)
    sys.exit(0)
print(child.pid, flush=True)
{then}
"""


def run_parent(*, flags: str, then: str) -> tuple[subprocess.Popen[str], int]:
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT.format(flags=flags, then=then)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    line = parent.stdout.readline().strip()
    if line in ("unbound", "refused", ""):
        parent.wait(timeout=10)
        pytest.skip(f"this environment's own job does not allow it ({line or 'no output'})")
    return parent, int(line)


def wait_until_gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while _process_alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def stop(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="the no-op path is the non-Windows one")
def test_binding_is_a_no_op_off_windows() -> None:
    assert jobs.bind_children_to_this_process() is False


@windows
def test_a_child_dies_when_the_app_is_killed_outright() -> None:
    parent, child = run_parent(flags="0", then="import time; time.sleep(120)")
    try:
        assert _process_alive(child)
        parent.kill()  # TerminateProcess: no cleanup code in the parent runs
        parent.wait(timeout=10)
        assert wait_until_gone(child), "the child outlived its killed parent"
    finally:
        stop(child)


@windows
def test_an_installer_started_with_breakaway_outlives_the_app() -> None:
    """The one child that must survive: an update's installer."""
    flags = "subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.DETACHED_PROCESS"
    parent, child = run_parent(flags=flags, then="import os; os._exit(0)")
    try:
        parent.wait(timeout=10)
        time.sleep(1.0)
        assert _process_alive(child), "the breakaway child was ended with its parent"
    finally:
        stop(child)
