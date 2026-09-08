"""HTTP routes and the realtime WebSocket protocol."""

from __future__ import annotations

import array
import math

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.domain.transcript import Segment, Transcript
from koe.text.script import Language

SAMPLE_RATE = 16_000


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Services.default()))


def tone(ms: float, amplitude: int = 8000) -> bytes:
    count = int(SAMPLE_RATE * ms / 1000.0)
    samples = array.array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * 220.0 * i / SAMPLE_RATE)) for i in range(count)),
    )
    return samples.tobytes()


def quiet(ms: float) -> bytes:
    count = int(SAMPLE_RATE * ms / 1000.0)
    return array.array("h", ((20 if i % 7 == 0 else -20) for i in range(count))).tobytes()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def test_health_reports_backend_and_providers(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["japanese_tokenizer"] in {"mecab", "character"}
    assert len(body["providers"]) == 2


def test_providers_endpoint_explains_the_ranking(client: TestClient) -> None:
    """Operators need to see why a backend was or was not chosen."""
    body = client.get("/v1/providers").json()
    names = [p["name"] for p in body["ranked"]]
    assert set(names) == {"mock-fast", "mock-accurate"}
    assert all("score" in p for p in body["ranked"])
    assert "breakers" in body


def test_transcribe_returns_diarized_segments(client: TestClient) -> None:
    body = client.post(
        "/v1/transcribe", json={"meeting": "quarterly-ja", "priority": "quality"}
    ).json()

    assert body["text"]
    assert body["language"] == "ja"
    assert body["segments"]
    assert all(s["speaker"] for s in body["segments"])


def test_priority_changes_which_backend_is_selected(client: TestClient) -> None:
    """The routing decision is observable through the API, not just internal."""
    quality = client.post("/v1/transcribe", json={"priority": "quality", "language": "ja"}).json()
    cost = client.post("/v1/transcribe", json={"priority": "cost", "language": "ja"}).json()

    assert quality["routed_to"] == "mock-accurate"
    assert cost["routed_to"] == "mock-fast"


def test_an_invalid_priority_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/transcribe", json={"priority": "nonsense"}).status_code >= 400


def test_minutes_endpoint_verifies_claims(client: TestClient) -> None:
    transcript = Transcript(
        segments=[
            Segment(
                text="金曜日までに対応します。",
                start=0,
                end=3,
                speaker="佐藤",
                language=Language.JA,
            )
        ],
        language=Language.JA,
    )
    body = client.post(
        "/v1/minutes", json={"transcript": transcript.model_dump(mode="json")}
    ).json()

    assert "minutes" in body
    assert "rendered" in body
    assert body["total_claims"] >= 0


def test_index_serves_the_client(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "koe" in response.text.lower()


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------


def test_stream_requires_a_start_frame_before_audio(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_bytes(tone(100))
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "start" in message["message"]


def test_stream_acknowledges_start_with_the_chosen_provider(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        message = ws.receive_json()
        assert message["type"] == "started"
        assert message["provider"] in {"mock-fast", "mock-accurate"}
        assert message["session_id"]


def test_a_full_streaming_session_yields_a_transcript(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja", "partial_interval_ms": 200})
        assert ws.receive_json()["type"] == "started"

        ws.send_bytes(quiet(300))
        for _ in range(6):
            ws.send_bytes(tone(200))
        ws.send_bytes(quiet(900))

        ws.send_json({"type": "stop"})

        # drain interim frames until the closing transcript arrives
        messages = []
        for _ in range(80):
            message = ws.receive_json()
            messages.append(message)
            if message["type"] == "transcript":
                break

        final = messages[-1]
        assert final["type"] == "transcript"
        assert final["duration"] > 0
        assert {m["type"] for m in messages} & {"speech", "partial", "final", "transcript"}


def test_oversized_audio_frames_are_rejected(client: TestClient) -> None:
    """A frame this size is not audio; it is someone probing the endpoint."""
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        assert ws.receive_json()["type"] == "started"

        ws.send_bytes(b"\x00" * (2 << 20))
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "too large" in message["message"]


def test_malformed_control_frames_do_not_kill_the_session(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"

        ws.send_json({"type": "start", "language": "en"})
        assert ws.receive_json()["type"] == "started"


def test_disconnecting_mid_utterance_is_clean(client: TestClient) -> None:
    """The normal way a voice session ends is the browser tab disappearing."""
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        assert ws.receive_json()["type"] == "started"
        ws.send_bytes(tone(400))
        # exit the context manager mid-speech, without a stop frame

    # the server must remain healthy for the next connection
    assert client.get("/health").json()["status"] == "ok"
