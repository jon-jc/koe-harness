"""Audio primitives for the streaming path.

These types sit on the hot path -- a 16 kHz session produces a chunk every
20-100 ms, for every concurrent call -- so they are frozen dataclasses with
``slots`` rather than pydantic models. Validation belongs at the system edge
(the websocket handler parses once), not on every frame; running a validator
per chunk would spend a measurable fraction of the latency budget re-checking
facts that cannot have changed since the last chunk.

Transcript and minutes types, which cross the API boundary far less often, do
use pydantic. The split is deliberate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class Encoding(StrEnum):
    """Wire encoding of an audio payload."""

    PCM_S16LE = "pcm_s16le"
    PCM_F32LE = "pcm_f32le"
    OPUS = "opus"
    MP3 = "mp3"
    WAV = "wav"

    @property
    def bytes_per_sample(self) -> int:
        """Bytes per sample per channel; 0 for compressed formats."""
        return {Encoding.PCM_S16LE: 2, Encoding.PCM_F32LE: 4}.get(self, 0)

    @property
    def is_pcm(self) -> bool:
        return self.bytes_per_sample > 0


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Sample rate, channel count and encoding of a stream.

    The default is 16 kHz mono PCM16 -- what essentially every ASR model wants,
    and what the browser client downsamples to before sending. Doing that
    conversion in the browser rather than the server is a deliberate cost
    decision: it moves the resampling onto the client's CPU and cuts uplink
    bandwidth by ~6x versus 48 kHz float.
    """

    sample_rate: int = 16_000
    channels: int = 1
    encoding: Encoding = Encoding.PCM_S16LE

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")

    @property
    def frame_size(self) -> int:
        """Bytes per frame (one sample across all channels)."""
        return self.encoding.bytes_per_sample * self.channels

    def duration_of(self, num_bytes: int) -> float:
        """Seconds of audio represented by `num_bytes`."""
        if not self.encoding.is_pcm:
            raise ValueError(f"cannot compute duration for compressed encoding {self.encoding}")
        return num_bytes / (self.frame_size * self.sample_rate)

    def bytes_for(self, seconds: float) -> int:
        """Bytes needed to hold `seconds` of audio, rounded to whole frames."""
        if not self.encoding.is_pcm:
            raise ValueError(f"cannot compute size for compressed encoding {self.encoding}")
        return math.floor(seconds * self.sample_rate) * self.frame_size

    def __str__(self) -> str:
        return f"{self.sample_rate}Hz/{self.channels}ch/{self.encoding.value}"


#: 16 kHz mono PCM16 -- the format ASR backends expect.
STANDARD_FORMAT = AudioFormat()


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A contiguous span of audio with its position in the stream.

    `offset` is seconds from the start of the session, which every downstream
    timestamp is expressed against. Carrying it on the chunk rather than
    recomputing from a running byte count means a dropped or reordered chunk
    cannot silently shift every subsequent word's timing.
    """

    data: bytes
    format: AudioFormat = STANDARD_FORMAT
    offset: float = 0.0
    sequence: int = 0

    @property
    def duration(self) -> float:
        """Length of this chunk in seconds."""
        return self.format.duration_of(len(self.data))

    @property
    def end(self) -> float:
        """Session-relative end time in seconds."""
        return self.offset + self.duration

    @property
    def num_samples(self) -> int:
        return len(self.data) // self.format.frame_size if self.format.frame_size else 0

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"<AudioChunk seq={self.sequence} {self.duration * 1000:.0f}ms "
            f"@{self.offset:.2f}s {len(self.data)}B>"
        )


@dataclass(slots=True)
class AudioBuffer:
    """Accumulates chunks and hands back fixed-size windows.

    ASR backends want windows of a particular length, but the network delivers
    whatever the client's encoder produced. This decouples the two: push
    arbitrary chunks, pull uniform windows.

    Mutable and explicitly **not** thread-safe -- one buffer belongs to one
    session's task, and sharing one across tasks is a bug this type will not
    hide from you.
    """

    format: AudioFormat = STANDARD_FORMAT
    _data: bytearray = field(default_factory=bytearray, repr=False)
    _consumed_bytes: int = 0
    _sequence: int = 0

    def push(self, chunk: AudioChunk | bytes) -> None:
        """Append audio to the buffer."""
        payload = chunk.data if isinstance(chunk, AudioChunk) else chunk
        self._data.extend(payload)

    @property
    def pending_bytes(self) -> int:
        return len(self._data)

    @property
    def pending_duration(self) -> float:
        return self.format.duration_of(len(self._data))

    @property
    def consumed_duration(self) -> float:
        """Seconds already handed out, i.e. the offset of the next window."""
        return self.format.duration_of(self._consumed_bytes)

    def take(self, seconds: float) -> AudioChunk | None:
        """Remove and return exactly `seconds` of audio, or ``None`` if short."""
        want = self.format.bytes_for(seconds)
        if want == 0 or len(self._data) < want:
            return None
        payload = bytes(self._data[:want])
        del self._data[:want]
        chunk = AudioChunk(
            data=payload,
            format=self.format,
            offset=self.consumed_duration,
            sequence=self._sequence,
        )
        self._consumed_bytes += want
        self._sequence += 1
        return chunk

    def take_all(self) -> AudioChunk | None:
        """Drain whatever is buffered, for flushing at end of stream."""
        if not self._data:
            return None
        payload = bytes(self._data)
        self._data.clear()
        chunk = AudioChunk(
            data=payload,
            format=self.format,
            offset=self.consumed_duration,
            sequence=self._sequence,
        )
        self._consumed_bytes += len(payload)
        self._sequence += 1
        return chunk

    def clear(self) -> None:
        """Drop buffered audio, keeping the stream position."""
        self._data.clear()
