"""The realtime streaming session.

One session per call leg. It owns the audio buffer, the endpointer, the
stabilizer and the transcript, and emits everything downstream as kernel events
so that plugins -- persistence, metrics, websocket broadcast, live translation --
attach without the session knowing they exist.

Events, in the order a normal utterance produces them::

    session.start   SessionInfo
    speech.start    SpeechEvent      -- endpointer detected onset
    asr.partial     PartialEvent     -- stabilized interim text (repeats)
    asr.final       Segment          -- utterance finished, text settled
    speech.end      SpeechEvent
    session.end     Transcript

Two things this deliberately does *not* do.

**It does not re-transcribe from the start of the session.** Interim
recognition runs only over the audio of the utterance in progress, and audio is
released once an utterance is final. A session that keeps every byte grows
without bound -- an hour of 16 kHz mono PCM16 is ~115 MB per concurrent
call -- and re-decoding the whole session for each partial makes cost grow
quadratically with meeting length.

**It does not block audio intake on recognition.** Pushing audio buffers and
returns; ASR runs as a task. If recognition falls behind, audio still arrives
and the endpointer keeps working, because dropping a caller's speech to wait on
a slow model is the one failure a voice product cannot recover from.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass

from koe.domain.audio import STANDARD_FORMAT, AudioChunk, AudioFormat
from koe.domain.transcript import Segment, Transcript
from koe.kernel.context import Context
from koe.pipeline.stabilizer import Stabilizer
from koe.pipeline.vad import VAD, SpeechSegment, VADConfig
from koe.providers.base import ASRProvider, ProviderError, Usage
from koe.text.script import Language, primary_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """How one session behaves."""

    language: Language = Language.UNKNOWN
    audio_format: AudioFormat = STANDARD_FORMAT
    #: How often interim recognition runs while someone is speaking. Lower
    #: feels more responsive and costs proportionally more, since every
    #: partial is a decode of the utterance so far.
    partial_interval_ms: float = 500.0
    vad: VADConfig | None = None
    emit_partials: bool = True

    def vad_config(self) -> VADConfig:
        return self.vad or VADConfig.for_language(self.language)


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    language: Language
    started_at: float


@dataclass(slots=True)
class SpeechEvent:
    session_id: str
    at: float
    state: str


@dataclass(slots=True)
class PartialEvent:
    """An interim hypothesis.

    `committed` will not change; `pending` may. Clients render them
    differently -- committed as normal text, pending greyed -- so a viewer can
    see what has settled without watching the whole line rewrite itself.
    """

    session_id: str
    committed: str
    pending: str
    language: Language
    start: float


class StreamingSession:
    """Drives one live audio stream to a transcript."""

    def __init__(
        self,
        ctx: Context,
        asr: ASRProvider,
        *,
        config: SessionConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.asr = asr
        self.config = config or SessionConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.started_at = time.time()

        self._format = self.config.audio_format
        self._vad = VAD(config=self.config.vad_config(), format=self._format)
        self._stabilizer = Stabilizer(language=self.config.language)

        # Audio for the utterance in progress only. Released at each endpoint.
        self._retained = bytearray()
        self._retained_start = 0.0
        self._position = 0.0

        self._segments: list[Segment] = []
        self._usage = Usage()
        self._last_partial_at = 0.0
        self._partial_task: asyncio.Task[None] | None = None
        self._closed = False
        self._sequence = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self.ctx.emit(
            "session.start",
            SessionInfo(
                session_id=self.session_id,
                language=self.config.language,
                started_at=self.started_at,
            ),
        )

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def duration(self) -> float:
        return self._position

    def transcript(self) -> Transcript:
        language = self.config.language
        if language is Language.UNKNOWN and self._segments:
            language = primary_language("".join(s.text for s in self._segments))
        return Transcript(
            segments=list(self._segments),
            language=language,
            duration=self._position,
            provider=self.asr.info.name,
            model=self.asr.info.model,
        )

    # -- audio ---------------------------------------------------------------

    async def push_audio(self, data: bytes | AudioChunk) -> None:
        """Feed audio into the session.

        Returns as soon as the audio is buffered. Recognition happens on its
        own task so that a slow model cannot stall intake.
        """
        if self._closed:
            raise RuntimeError(f"session {self.session_id} is closed")

        payload = data.data if isinstance(data, AudioChunk) else data
        if not payload:
            return

        self._retained.extend(payload)
        self._position += self._format.duration_of(len(payload))

        was_speaking = self._vad.in_speech
        finished = self._vad.push(payload)

        if not was_speaking and self._vad.in_speech:
            await self.ctx.emit(
                "speech.start",
                SpeechEvent(session_id=self.session_id, at=self._position, state="start"),
            )

        for segment in finished:
            await self._finalize(segment)

        if self.config.emit_partials and self._vad.in_speech and self._should_emit_partial():
            self._schedule_partial()

    def _should_emit_partial(self) -> bool:
        elapsed_ms = (self._position - self._last_partial_at) * 1000.0
        return elapsed_ms >= self.config.partial_interval_ms

    def _schedule_partial(self) -> None:
        # One partial in flight at a time. Queueing them would spend money and
        # latency producing hypotheses that are already stale on arrival.
        if self._partial_task is not None and not self._partial_task.done():
            return
        self._last_partial_at = self._position
        self._partial_task = asyncio.create_task(
            self._recognize_partial(), name=f"koe.partial.{self.session_id}"
        )

    def _utterance_audio(self, start: float, end: float) -> AudioChunk | None:
        """Slice retained audio for ``[start, end]``."""
        offset_start = max(0, self._format.bytes_for(start - self._retained_start))
        offset_end = self._format.bytes_for(end - self._retained_start)
        payload = bytes(self._retained[offset_start:offset_end])
        if not payload:
            return None
        return AudioChunk(data=payload, format=self._format, offset=start)

    # -- recognition ---------------------------------------------------------

    async def _recognize_partial(self) -> None:
        chunk = self._utterance_audio(self._retained_start, self._position)
        if chunk is None:
            return
        try:
            result = await self.asr.transcribe(chunk, language=self.config.language or None)
        except ProviderError as exc:
            # A failed partial is cosmetic: the final pass still runs over the
            # same audio, so this must never surface as a session error.
            logger.debug("partial recognition failed for %s: %s", self.session_id, exc)
            return
        except asyncio.CancelledError:
            raise

        text = result.text
        if not text:
            return

        stabilized = self._stabilizer.update(text)
        await self.ctx.emit(
            "asr.partial",
            PartialEvent(
                session_id=self.session_id,
                committed=stabilized.committed,
                pending=stabilized.pending,
                language=stabilized.language,
                start=self._retained_start,
            ),
        )

    async def _finalize(self, speech: SpeechSegment) -> None:
        await self._cancel_partial()

        chunk = self._utterance_audio(speech.start, speech.end)
        if chunk is not None:
            try:
                result = await self.asr.transcribe(chunk, language=self.config.language or None)
            except ProviderError as exc:
                logger.warning("recognition failed for %s: %s", self.session_id, exc)
                result = None
            else:
                self._usage = self._usage + Usage(
                    provider=self.asr.info.name,
                    model=self.asr.info.model,
                    audio_seconds=chunk.duration,
                    cost_usd=self.asr.info.estimate_audio_cost(chunk.duration),
                )

            if result is not None and result.text:
                stabilized = self._stabilizer.finalize(result.text)
                # Carry through a speaker label if the backend produced one.
                # Fused ASR+diarization backends attribute segments themselves,
                # and dropping that here would silently discard the answer to
                # "who said this" on every provider that already knows.
                speaker = next((seg.speaker for seg in result.segments if seg.speaker), None)
                segment = Segment(
                    text=stabilized.committed,
                    start=speech.start,
                    end=speech.end,
                    language=stabilized.language,
                    speaker=speaker,
                    is_final=True,
                )
                self._segments.append(segment)
                self._sequence += 1
                await self.ctx.emit("asr.final", segment)

        await self.ctx.emit(
            "speech.end",
            SpeechEvent(session_id=self.session_id, at=speech.end, state="end"),
        )

        # Release audio for the finished utterance; a session that keeps
        # everything grows without bound.
        self._stabilizer.reset()
        consumed = self._format.bytes_for(speech.end - self._retained_start)
        if consumed > 0:
            del self._retained[:consumed]
            self._retained_start = speech.end

    async def _cancel_partial(self) -> None:
        task, self._partial_task = self._partial_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # -- shutdown ------------------------------------------------------------

    async def finish(self) -> Transcript:
        """Flush any open utterance and close the session.

        The flush matters: without it, whatever the caller said immediately
        before hanging up is lost, which is most calls.
        """
        if self._closed:
            return self.transcript()
        self._closed = True

        await self._cancel_partial()

        trailing = self._vad.flush()
        if trailing is not None:
            await self._finalize(trailing)

        transcript = self.transcript()
        await self.ctx.emit("session.end", transcript)
        return transcript

    async def __aenter__(self) -> StreamingSession:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.finish()
