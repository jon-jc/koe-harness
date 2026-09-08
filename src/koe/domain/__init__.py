"""Domain types: audio on the hot path, transcripts at the API boundary.

The two halves use different tools on purpose. Audio types are frozen
dataclasses with ``slots`` because they are constructed per frame, per
concurrent session; transcript types are pydantic models because they are
serialized, persisted and validated at the system edge, where correctness
matters more than allocation count.
"""

from koe.domain.audio import (
    STANDARD_FORMAT,
    AudioBuffer,
    AudioChunk,
    AudioFormat,
    Encoding,
)
from koe.domain.transcript import (
    UNKNOWN_SPEAKER,
    Diarization,
    Segment,
    SpeakerTurn,
    Transcript,
    Word,
    attribute_speakers,
)

__all__ = [
    "STANDARD_FORMAT",
    "UNKNOWN_SPEAKER",
    "AudioBuffer",
    "AudioChunk",
    "AudioFormat",
    "Diarization",
    "Encoding",
    "Segment",
    "SpeakerTurn",
    "Transcript",
    "Word",
    "attribute_speakers",
]
