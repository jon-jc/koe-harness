"""Server-driven demo playback.

The demo is not decoration — it is the only way most people will see the
realtime path, so it needs to actually exercise it. These tests pin that it
drives the real session rather than replaying events, and that the re-timing it
depends on is correct.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from koe.api.app import Services, create_app
from koe.api.demo import drive_demo, retime
from koe.config import Settings
from koe.kernel.context import Context
from koe.pipeline.session import SessionConfig, StreamingSession
from koe.providers.mock import MEETING_JA, MockASR
from koe.text.script import Language


def demo_session(script: list) -> tuple[Context, StreamingSession]:
    ctx = Context()
    asr = MockASR(script=script, degradation=0.0, timeline=True, name="demo-asr")
    session = StreamingSession(ctx, asr, config=SessionConfig(language=Language.JA))
    return ctx, session


# --------------------------------------------------------------------------
# re-timing
# --------------------------------------------------------------------------


def test_retiming_opens_gaps_the_endpointer_can_act_on() -> None:
    """The corpus packs utterances 0.2s apart; JA endpointing needs 900ms."""
    original = MEETING_JA
    assert original[1].start - original[0].end < 0.5  # the problem

    retimed = retime(original)

    gaps = [b.start - a.end for a, b in zip(retimed, retimed[1:], strict=False)]
    assert all(gap >= 0.9 for gap in gaps), gaps


def test_retiming_preserves_content_and_order() -> None:
    retimed = retime(MEETING_JA)
    assert [u.text for u in retimed] == [u.text for u in MEETING_JA]
    assert [u.speaker for u in retimed] == [u.speaker for u in MEETING_JA]
    assert retimed == sorted(retimed, key=lambda u: u.start)


def test_retiming_starts_after_a_beat_of_room_tone() -> None:
    """The noise floor needs a moment to settle before the first onset."""
    assert retime(MEETING_JA)[0].start > 0.0


# --------------------------------------------------------------------------
# playback
# --------------------------------------------------------------------------


async def test_playback_produces_one_segment_per_utterance() -> None:
    """End to end through the real VAD, stabilizer and ASR."""
    script = retime(MEETING_JA)
    _, session = demo_session(script)
    await session.start()

    await drive_demo(session, script, speed=60.0)
    transcript = await session.finish()

    assert len(transcript.final_segments) == len(script)
    assert [s.text for s in transcript.final_segments] == [u.text for u in script]


async def test_playback_attributes_speakers() -> None:
    """A fused backend reports speakers; the session must carry them through."""
    script = retime(MEETING_JA)
    _, session = demo_session(script)
    await session.start()

    await drive_demo(session, script, speed=60.0)
    transcript = await session.finish()

    assert [s.speaker for s in transcript.final_segments] == [u.speaker for u in script]


async def test_playback_reports_the_level_it_generated() -> None:
    """The client has no microphone during a demo; a dead meter reads as broken."""
    script = retime(MEETING_JA[:2])
    _, session = demo_session(script)
    await session.start()

    levels: list[float] = []
    await drive_demo(session, script, speed=60.0, on_level=levels.append)
    await session.finish()

    assert levels
    assert max(levels) > 0.1  # speech
    assert min(levels) < 0.1  # silence between utterances


async def test_playback_paces_against_a_schedule() -> None:
    """A fixed per-chunk sleep drifts slower the busier the pipeline gets."""
    script = retime(MEETING_JA[:3])
    audio_seconds = max(u.end for u in script) + 1.0
    _, session = demo_session(script)
    await session.start()

    started = asyncio.get_running_loop().time()
    await drive_demo(session, script, speed=20.0)
    elapsed = asyncio.get_running_loop().time() - started
    await session.finish()

    # Generous bound: the point is that pipeline time is absorbed by the
    # schedule rather than added to it, not that timing is exact.
    assert elapsed < audio_seconds / 20.0 * 4 + 1.0


async def test_an_empty_script_is_a_no_op() -> None:
    _, session = demo_session([])
    await session.start()
    await drive_demo(session, [], speed=60.0)
    transcript = await session.finish()
    assert transcript.final_segments == []


# --------------------------------------------------------------------------
# over the websocket
# --------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Services.default(Settings(environment="local"))))


def test_demo_frame_starts_a_session(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "demo", "meeting": "quarterly-ja", "language": "ja"})
        message = ws.receive_json()
        assert message["type"] == "started"
        assert message["provider"] == "demo-asr"


def test_an_unknown_meeting_is_rejected(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "demo", "meeting": "does-not-exist"})
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "unknown" in message["message"]


def test_demo_yields_transcript_then_finished(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "demo", "meeting": "quarterly-ja", "language": "ja"})
        assert ws.receive_json()["type"] == "started"

        seen: list[str] = []
        for _ in range(4000):
            seen.append(ws.receive_json()["type"])
            if seen[-1] == "demo_finished":
                break

        assert "demo_finished" in seen
        assert "transcript" in seen
        assert "final" in seen
        assert "level" in seen


def test_minutes_can_be_requested_after_a_demo(client: TestClient) -> None:
    """The transcript arrived on this socket; the client should not re-upload it."""
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "demo", "meeting": "quarterly-ja", "language": "ja"})
        for _ in range(4000):
            if ws.receive_json()["type"] == "demo_finished":
                break

        ws.send_json({"type": "minutes"})
        for _ in range(50):
            message = ws.receive_json()
            if message["type"] == "minutes":
                break

        assert message["type"] == "minutes"
        assert message["total_claims"] > 0
        # The canned minutes plant one fabricated decision so the guardrail is
        # observable; it must be dropped rather than published.
        assert message["dropped"]


def test_minutes_without_a_transcript_is_refused(client: TestClient) -> None:
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "minutes"})
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "transcript" in message["message"]
