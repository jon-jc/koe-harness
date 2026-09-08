"""Where the API looks for the built web client.

This has its own module because getting it wrong is uniquely hard to notice.
The desktop application shipped for two releases serving the "build the client"
placeholder instead of the app: `/health` was fine, `/v1/*` was fine, the
websocket was fine, and the packaging check asked all three. The only symptom
was a window with a stub page in it — which nothing automated was looking at.

So the two path shapes are pinned here, and `packaging/build.py` now fetches
`/` from the frozen binary rather than trusting that a live server is a
correct one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from koe.api.app import _web_root, create_app


def test_a_checkout_resolves_to_the_repository_web_directory() -> None:
    root = _web_root()
    assert root.name == "web"
    # index.html is committed; dist/ is a build output that may not exist yet.
    assert (root / "index.html").is_file()


def test_a_frozen_bundle_resolves_inside_the_extraction_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyInstaller reports resources under `sys._MEIPASS`, not beside the exe.

    Walking up from `__file__` lands one level past the extraction directory in
    a bundle, on the application folder — where nothing is ever installed.
    """
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)
    assert _web_root() == tmp_path / "web"


def test_the_client_is_served_rather_than_the_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle laid out the way PyInstaller lays one out must serve the app."""
    web = tmp_path / "web"
    (web / "dist").mkdir(parents=True)
    (web / "dist" / "app.js").write_text("/* bundle */", encoding="utf-8")
    (web / "index.html").write_text("<h1>koe client</h1>", encoding="utf-8")
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

    with TestClient(create_app()) as client:
        body = client.get("/").text

    assert "koe client" in body
    assert "Build the client" not in body


def test_the_placeholder_still_appears_when_there_is_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback is correct behaviour for a checkout that has not run npm.

    It is only wrong when a build *did* happen and the path was miscomputed,
    which is what the test above covers.
    """
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Build the client" in response.text
