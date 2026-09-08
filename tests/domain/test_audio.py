"""Audio format arithmetic and the streaming buffer."""

from __future__ import annotations

import pytest

from koe.domain.audio import STANDARD_FORMAT, AudioBuffer, AudioChunk, AudioFormat, Encoding


def pcm(seconds: float, fmt: AudioFormat = STANDARD_FORMAT) -> bytes:
    return b"\x00" * fmt.bytes_for(seconds)


def test_standard_format_is_16k_mono_pcm16() -> None:
    """What essentially every ASR model wants, and what the client sends."""
    assert STANDARD_FORMAT.sample_rate == 16_000
    assert STANDARD_FORMAT.channels == 1
    assert STANDARD_FORMAT.frame_size == 2


def test_invalid_formats_are_rejected() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        AudioFormat(sample_rate=0)
    with pytest.raises(ValueError, match="channels"):
        AudioFormat(channels=0)


def test_duration_and_size_round_trip() -> None:
    assert STANDARD_FORMAT.duration_of(32_000) == 1.0
    assert STANDARD_FORMAT.bytes_for(1.0) == 32_000
    assert STANDARD_FORMAT.bytes_for(0.5) == 16_000


def test_stereo_doubles_the_frame_size() -> None:
    stereo = AudioFormat(channels=2)
    assert stereo.frame_size == 4
    assert stereo.duration_of(64_000) == 1.0


def test_compressed_encodings_reject_duration_arithmetic() -> None:
    """Guessing a duration for Opus would silently corrupt every timestamp."""
    opus = AudioFormat(encoding=Encoding.OPUS)
    with pytest.raises(ValueError, match="compressed"):
        opus.duration_of(1000)
    with pytest.raises(ValueError, match="compressed"):
        opus.bytes_for(1.0)


def test_chunk_reports_its_place_in_the_stream() -> None:
    chunk = AudioChunk(data=pcm(0.5), offset=2.0, sequence=4)
    assert chunk.duration == 0.5
    assert chunk.end == 2.5
    assert chunk.num_samples == 8_000
    assert len(chunk) == 16_000


# --------------------------------------------------------------------------
# buffer
# --------------------------------------------------------------------------


def test_buffer_hands_back_uniform_windows_from_ragged_input() -> None:
    """The network delivers whatever the encoder produced; ASR wants fixed windows."""
    buffer = AudioBuffer()
    buffer.push(pcm(0.3))
    buffer.push(pcm(0.3))
    buffer.push(pcm(0.6))

    first = buffer.take(0.5)
    second = buffer.take(0.5)

    assert first is not None and second is not None
    assert first.duration == pytest.approx(0.5)
    assert second.duration == pytest.approx(0.5)
    assert buffer.pending_duration == pytest.approx(0.2)


def test_take_returns_none_when_short() -> None:
    buffer = AudioBuffer()
    buffer.push(pcm(0.1))
    assert buffer.take(0.5) is None
    assert buffer.pending_duration == pytest.approx(0.1)


def test_offsets_advance_monotonically_across_windows() -> None:
    """Every downstream timestamp is expressed against these offsets."""
    buffer = AudioBuffer()
    buffer.push(pcm(2.0))

    offsets = []
    while (chunk := buffer.take(0.5)) is not None:
        offsets.append(chunk.offset)

    assert offsets == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_sequence_numbers_increment() -> None:
    buffer = AudioBuffer()
    buffer.push(pcm(1.0))
    chunks = [buffer.take(0.5), buffer.take(0.5)]
    assert [c.sequence for c in chunks if c] == [0, 1]


def test_take_all_drains_the_tail() -> None:
    """End of stream: whatever is left still has to be transcribed."""
    buffer = AudioBuffer()
    buffer.push(pcm(0.7))
    buffer.take(0.5)

    tail = buffer.take_all()

    assert tail is not None
    assert tail.duration == pytest.approx(0.2)
    assert tail.offset == pytest.approx(0.5)
    assert buffer.take_all() is None


def test_clear_drops_audio_but_keeps_stream_position() -> None:
    buffer = AudioBuffer()
    buffer.push(pcm(1.0))
    buffer.take(0.5)
    buffer.clear()
    buffer.push(pcm(0.5))

    chunk = buffer.take(0.5)

    assert chunk is not None
    assert chunk.offset == pytest.approx(0.5)


def test_buffer_accepts_raw_bytes_and_chunks() -> None:
    buffer = AudioBuffer()
    buffer.push(pcm(0.5))
    buffer.push(AudioChunk(data=pcm(0.5)))
    assert buffer.pending_duration == pytest.approx(1.0)
