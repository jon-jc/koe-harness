"""The plugins, tools and workspace endpoints.

These endpoints expose a filesystem and an execution path over HTTP, so the
tests that matter most are the refusals: a path outside the workspace, a tool
that does not exist, a plugin name nobody registered.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.config import Settings
from koe.workspace import Workspace


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    workspace = tmp_path / "project"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "main.py").write_text("print('議事録')\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# notes\n", encoding="utf-8")

    services = Services.default(Settings(environment="local"))
    services.workspace = Workspace(workspace)
    # Re-provide, because the plugins already mounted against the default one.
    services.ctx.provide("workspace", services.workspace, replace=True)
    return TestClient(create_app(services))


# --------------------------------------------------------------------------
# plugins
# --------------------------------------------------------------------------


def test_the_builtin_tool_groups_are_listed_as_plugins(client: TestClient) -> None:
    """First-party tools are plugins, or the plugin path is decoration."""
    body = client.get("/v1/plugins").json()
    names = {p["name"] for p in body["plugins"]}
    assert {"workspace-tools", "meeting-tools"} <= names
    assert all(p["builtin"] for p in body["plugins"])


def test_disabling_a_plugin_removes_its_tools(client: TestClient) -> None:
    assert any(t["name"] == "read_file" for t in client.get("/v1/tools").json()["tools"])

    body = client.put("/v1/plugins/workspace-tools", json={"enabled": False}).json()

    assert body["plugin"]["active"] is False
    assert not any(t["name"] == "read_file" for t in body["tools"])
    assert not any(t["name"] == "read_file" for t in client.get("/v1/tools").json()["tools"])


def test_re_enabling_restores_them(client: TestClient) -> None:
    client.put("/v1/plugins/workspace-tools", json={"enabled": False})
    client.put("/v1/plugins/workspace-tools", json={"enabled": True})

    assert any(t["name"] == "read_file" for t in client.get("/v1/tools").json()["tools"])


def test_an_unknown_plugin_is_a_404(client: TestClient) -> None:
    assert client.put("/v1/plugins/nonesuch", json={"enabled": True}).status_code == 404


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_the_tool_listing_carries_host_fields(client: TestClient) -> None:
    """This endpoint feeds a settings panel, not a model request."""
    tools = client.get("/v1/tools").json()["tools"]
    grep = next(t for t in tools if t["name"] == "grep_files")
    assert grep["source"] == "workspace"
    assert grep["timeout_s"] == 20.0


def test_calling_a_tool_directly(client: TestClient) -> None:
    body = client.post("/v1/tools/list_files", json={"arguments": {}}).json()
    assert body["ok"]
    assert "notes.md" in body["content"]


def test_an_unknown_tool_is_a_result_not_a_500(client: TestClient) -> None:
    """The pipeline never raises; a caller gets a routable failure."""
    response = client.post("/v1/tools/nonesuch", json={"arguments": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "no_tool"


def test_a_tool_cannot_escape_the_workspace(client: TestClient) -> None:
    body = client.post(
        "/v1/tools/read_file", json={"arguments": {"path": "../../../etc/passwd"}}
    ).json()
    assert body["ok"] is False
    assert "outside the workspace" in body["detail"]


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------


def test_the_tree_lists_the_workspace(client: TestClient) -> None:
    body = client.get("/v1/fs/tree").json()
    names = {entry["name"] for entry in body["entries"]}
    assert names == {"src", "notes.md"}
    assert body["name"] == "project"


def test_reading_a_file_returns_its_language(client: TestClient) -> None:
    body = client.get("/v1/fs/file", params={"path": "src/main.py"}).json()
    assert body["language"] == "python"
    assert "議事録" in body["text"]


def test_a_missing_file_is_a_404(client: TestClient) -> None:
    assert client.get("/v1/fs/file", params={"path": "gone.py"}).status_code == 404


def test_a_path_outside_the_workspace_is_a_400(client: TestClient) -> None:
    """Not a 404: the file may well exist. We are refusing, not failing to find."""
    response = client.get("/v1/fs/file", params={"path": "../../../etc/passwd"})
    assert response.status_code == 400
    assert "outside the workspace" in response.json()["detail"]


def test_search_returns_hits_with_line_numbers(client: TestClient) -> None:
    body = client.get("/v1/fs/search", params={"q": "議事録"}).json()
    assert body["hits"][0]["path"] == "src/main.py"
    assert body["hits"][0]["line"] == 1


def test_an_invalid_search_pattern_is_a_400(client: TestClient) -> None:
    assert client.get("/v1/fs/search", params={"q": "(unclosed"}).status_code == 400
