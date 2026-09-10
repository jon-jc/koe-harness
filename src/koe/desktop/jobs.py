"""Tie koe's child processes to koe's own lifetime.

A terminal in koe is a real ConPTY session: ``OpenConsole.exe`` and a shell,
started as children of the app. Windows does not end a process's children
when it exits, so an app that is *killed* rather than closed — a crash, Task
Manager, a build script's verification step — leaves them running. They are
disposed on a clean shutdown, which is exactly why the gap went unnoticed.

Found by accumulation: ten orphaned ``OpenConsole.exe`` processes over two
days of development, one of them holding a file inside the build folder so the
next build could not replace it. An update has to replace that same file, and
would fail the same way on a machine where koe had ever been killed with a
terminal open.

So at startup the app places itself in a job object with
``KILL_ON_JOB_CLOSE``. Every process it starts inherits the job, and when the
last handle to the job closes — which the OS does as the process exits,
however it exits — they are ended with it.

``BREAKAWAY_OK`` is set too, for the one child that must outlive koe: the
installer an update starts, which waits for koe to exit and then replaces it.
Without it, installing an update would kill its own installer.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
#: ``JobObjectExtendedLimitInformation`` in the JOBOBJECTINFOCLASS enumeration.
EXTENDED_LIMIT_INFORMATION = 9

#: Held for the life of the process and deliberately never closed: closing it
#: is what ends the children.
_job: int | None = None


def bind_children_to_this_process() -> bool:
    """Put this process in a kill-on-close job. True if it now is.

    Safe to call more than once, and a no-op off Windows, where the terminal's
    children are in their own session and a dead parent's pty closes anyway.
    """
    global _job
    if sys.platform != "win32":
        return False
    if _job is not None:
        return True

    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("could not create a job object (error %d)", ctypes.get_last_error())
        return False

    info = ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
    )
    if not kernel32.SetInformationJobObject(
        job, EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        logger.warning("could not configure the job object (error %d)", ctypes.get_last_error())
        kernel32.CloseHandle(job)
        return False

    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        # Already inside a job that does not allow this one. Children then live
        # as they always did, which is a leak on a hard kill and not a failure.
        logger.info(
            "could not join a job object (error %d); child processes are not bound",
            ctypes.get_last_error(),
        )
        kernel32.CloseHandle(job)
        return False

    _job = int(job)
    logger.info("child processes are bound to this process's lifetime")
    return True
