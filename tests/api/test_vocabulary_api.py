"""The user vocabulary: its store, its plugin, and the endpoints that edit it.

The rules themselves are tested in `tests/text/test_dictation.py`. What matters
here is the wiring: that the word list survives a round trip through the file,
that an edit takes effect without anything re-registering, and that turning the
plugin off degrades to the recognizer's own output rather than to an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.config import Settings
from koe.pipeline.vocabulary import VocabularyStore, user_vocabulary


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    services = Services.default(Settings(environment="local"))
    # Point the mounted plugin at a temp file rather than the real config
    # directory: a test that edits the developer's own word list is a test
    # nobody runs twice.
    services.ctx.provide("vocabulary", VocabularyStore(tmp_path / "vocabulary.txt"), replace=True)
    return TestClient(create_app(services))


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def test_a_missing_file_is_an_empty_vocabulary_not_an_error(tmp_path: Path) -> None:
    """Nobody has written a word list yet, which is the normal first run."""
    store = VocabularyStore(tmp_path / "nothing-here.txt")
    assert len(store) == 0
    assert store.apply("unchanged") == "unchanged"


def test_saving_writes_the_file_and_recompiles(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.txt"
    store = VocabularyStore(path)

    store.save("Coe => koe\n山本")

    assert store.apply("the Coe harness") == "the koe harness"
    assert "Coe => koe" in path.read_text(encoding="utf-8")


def test_the_file_is_written_in_the_canonical_format(tmp_path: Path) -> None:
    """So that the file and the editor agree about what was saved."""
    store = VocabularyStore(tmp_path / "v.txt")
    store.save("  Coe   ->   koe  ")
    assert (tmp_path / "v.txt").read_text(encoding="utf-8") == "Coe => koe\n"


def test_a_reload_picks_up_an_edit_made_outside_the_app(tmp_path: Path) -> None:
    """It is a text file in a config directory; people will edit it directly."""
    path = tmp_path / "v.txt"
    path.write_text("Coe => koe\n", encoding="utf-8")
    store = VocabularyStore(path)
    assert store.apply("Coe") == "koe"

    path.write_text("Coe => KOE\n", encoding="utf-8")
    store.reload()
    assert store.apply("Coe") == "KOE"


def test_an_unreadable_file_does_not_take_the_session_with_it(tmp_path: Path) -> None:
    """The cost of ignoring a broken word list is some misrecognized nouns.
    The cost of raising here is the whole meeting."""
    directory = tmp_path / "vocabulary.txt"
    directory.mkdir()  # a directory where a file was expected

    store = VocabularyStore(directory)

    assert len(store) == 0
    assert store.apply("still works") == "still works"


def test_the_plugin_provides_the_service(tmp_path: Path) -> None:
    class FakeContext:
        def __init__(self) -> None:
            self.services: dict[str, object] = {}

        def provide(self, name: str, value: object, replace: bool = False) -> None:
            self.services[name] = value

    class Config:
        path = tmp_path / "v.txt"

    Config.path.write_text("Coe => koe\n", encoding="utf-8")
    ctx = FakeContext()
    user_vocabulary(ctx, Config())

    store = ctx.services["vocabulary"]
    assert isinstance(store, VocabularyStore)
    assert store.apply("Coe") == "koe"


# --------------------------------------------------------------------------
# the endpoints
# --------------------------------------------------------------------------


def test_the_word_list_starts_empty_and_enabled(client: TestClient) -> None:
    body = client.get("/v1/vocabulary").json()
    assert body["enabled"] is True
    assert body["entries"] == 0
    assert body["text"] == ""


def test_a_saved_list_comes_back(client: TestClient) -> None:
    saved = client.put("/v1/vocabulary", json={"text": "Coe => koe\n山本"}).json()
    assert saved["entries"] == 2
    assert saved["terms"] == ["koe", "山本"]

    assert client.get("/v1/vocabulary").json()["text"] == "Coe => koe\n山本"


def test_saving_an_empty_list_clears_it(client: TestClient) -> None:
    client.put("/v1/vocabulary", json={"text": "Coe => koe"})
    assert client.put("/v1/vocabulary", json={"text": ""}).json()["entries"] == 0


def test_editing_is_refused_when_the_plugin_is_off(client: TestClient) -> None:
    """409 rather than 404: a missing endpoint and a disabled feature are
    different things, and the panel says so differently."""
    app = client.app
    app.state.services.ctx.provide("vocabulary", None, replace=True)

    assert client.put("/v1/vocabulary", json={"text": "x"}).status_code == 409

    body = client.get("/v1/vocabulary").json()
    assert body["enabled"] is False
    assert body["entries"] == 0
