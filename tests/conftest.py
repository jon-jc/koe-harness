"""Suite-wide setup.

Set before anything imports koe, because `Settings` reads the environment at
construction and several modules build one at import time.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

# Discovery probes the well-known local-model ports at startup. Left on, the
# suite's behaviour would depend on whether the machine running it happens to
# have Ollama open -- tests that assert the mock LLM answers would get a real
# local model on a developer's laptop and the mock on CI. An explicit
# `local_llm_base_url` is still honoured, which is how the local tests
# exercise the path deliberately.
os.environ.setdefault("KOE_DISCOVER_LOCAL_MODELS", "false")


# koe reads and *writes* per-user state: which plugins are disabled, the
# vocabulary word list, stored credentials. Left alone, the suite writes to the
# developer's real configuration directory -- a test that disables a plugin
# leaves it disabled in their actual koe, and the next test sees the state the
# previous one persisted rather than a default.
#
# Both halves of that are real: it happened, and it produced a test that passed
# alone and failed in a suite. Redirected here rather than per-test, because a
# test that forgets is exactly the one that causes it.
_SANDBOX = pathlib.Path(tempfile.mkdtemp(prefix="koe-test-config-"))
os.environ.setdefault("APPDATA", str(_SANDBOX))
os.environ.setdefault("XDG_CONFIG_HOME", str(_SANDBOX))
os.environ.setdefault("XDG_DATA_HOME", str(_SANDBOX))


@pytest.fixture(autouse=True)
def _isolated_user_state(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Give every test its own config and data directories.

    A test *about* those functions opts out with `@pytest.mark.real_user_paths`
    -- explicitly, rather than this fixture guessing from the module name,
    because the one test that needs the real answer should be the one that says
    so.
    """
    if request.node.get_closest_marker("real_user_paths"):
        return

    from koe.desktop import paths

    monkeypatch.setattr(paths, "config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path / "data")
