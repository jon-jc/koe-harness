"""Whisper, on this machine, through faster-whisper.

This is the provider that makes koe's privacy claim literal. Everything else in
the local package keeps *text* on the device; this keeps the audio. For a
harness whose stated job is recording other people's meetings, that is the
difference between a tool a company can approve and one it cannot.

**In-process, not a server.** The local LLM path talks HTTP to something the
user started, because that is how llama.cpp and its descendants are deployed.
Whisper is a Python library, so koe loads it directly. There is no port, no
second process to supervise, and nothing to leave running.

**Model size is a quality decision, and in Japanese it is a correctness one.**
The sizes below carry per-language priors rather than a single number, because
the gap between them is not uniform: ``tiny`` and ``base`` produce English that
is rough but usable, and Japanese that is not. Whisper's training mix is
overwhelmingly English, and the smaller checkpoints spend what multilingual
capacity they have on languages closer to it. A size list that presents these as
a neutral speed/accuracy slider would be misleading in exactly the case koe
exists for, so :func:`usable_for` answers per language and the UI marks the ones
that are not.

**Loading is slow and happens once.** A cold ``large-v3`` is gigabytes off disk;
it is loaded lazily on the first transcription and held. The load runs on a
worker thread, as does every transcription, because CTranslate2 releases the GIL
but the Python wrapper around it does not.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from koe.domain.audio import AudioChunk
from koe.domain.transcript import Segment, Transcript, Word
from koe.providers.base import Modality, ProviderError, ProviderInfo
from koe.text.script import Language

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WhisperSize:
    """One checkpoint, and what it is honestly good for."""

    id: str
    label: str
    #: Rough on-disk size of the CTranslate2 conversion, for the UI to show
    #: before someone commits to a download on a metered connection.
    download_mb: int
    #: Real-time factor on a modern CPU: 0.5 means a minute of audio takes
    #: thirty seconds. The single most important number for a realtime tool,
    #: and the reason the large models are not a default.
    typical_rtf: float
    #: Per-language priors. Japanese is materially worse at every size, and
    #: catastrophically so at the small ones.
    error_rate: dict[Language, float]
    #: Languages this size is fit to be *offered* for. A size missing from here
    #: is still selectable, but koe says it is not suitable rather than letting
    #: someone discover it from a meeting transcript.
    suitable: frozenset[Language]


#: The checkpoints faster-whisper can fetch by name. `large-v3` is the one to
#: use for Japanese if the machine can carry it.
SIZES: tuple[WhisperSize, ...] = (
    WhisperSize(
        id="tiny",
        label="tiny",
        download_mb=75,
        typical_rtf=0.06,
        error_rate={Language.EN: 0.18, Language.JA: 0.55},
        suitable=frozenset(),
    ),
    WhisperSize(
        id="base",
        label="base",
        download_mb=145,
        typical_rtf=0.10,
        error_rate={Language.EN: 0.14, Language.JA: 0.42},
        suitable=frozenset(),
    ),
    WhisperSize(
        id="small",
        label="small",
        download_mb=480,
        typical_rtf=0.25,
        error_rate={Language.EN: 0.10, Language.JA: 0.26},
        suitable=frozenset({Language.EN}),
    ),
    WhisperSize(
        id="medium",
        label="medium",
        download_mb=1_530,
        typical_rtf=0.55,
        error_rate={Language.EN: 0.08, Language.JA: 0.16},
        suitable=frozenset({Language.EN, Language.JA}),
    ),
    WhisperSize(
        id="large-v3",
        label="large-v3",
        download_mb=3_090,
        typical_rtf=1.10,
        error_rate={Language.EN: 0.06, Language.JA: 0.10},
        suitable=frozenset({Language.EN, Language.JA}),
    ),
    WhisperSize(
        id="large-v3-turbo",
        label="large-v3-turbo",
        download_mb=1_620,
        typical_rtf=0.30,
        error_rate={Language.EN: 0.07, Language.JA: 0.12},
        suitable=frozenset({Language.EN, Language.JA}),
    ),
)

SIZES_BY_ID = {size.id: size for size in SIZES}

#: What koe picks when nobody has chosen. Turbo rather than large-v3: it is
#: within a point or two on Japanese at a third of the compute, which is the
#: difference between usable on a laptop and not.
DEFAULT_SIZE = "large-v3-turbo"


def usable_for(size_id: str, language: Language) -> bool:
    """Whether koe is willing to recommend this size for this language."""
    size = SIZES_BY_ID.get(size_id)
    if size is None:
        return False
    if language in (Language.UNKNOWN, Language.MIXED):
        return {Language.JA, Language.EN} <= size.suitable
    return language in size.suitable


def available() -> bool:
    """Whether faster-whisper can be imported, without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def _whisper_language(language: Language | None) -> str | None:
    """koe's language enum as Whisper's code.

    None means "detect", which is what MIXED and UNKNOWN want: a code-switched
    utterance forced to one language comes back transliterated into that
    language's script, which is worse than either answer alone.
    """
    if language is Language.JA:
        return "ja"
    if language is Language.EN:
        return "en"
    return None


@dataclass(slots=True)
class LocalWhisperASR:
    """Whisper running in this process, via faster-whisper."""

    size: str = DEFAULT_SIZE
    #: "cpu", "cuda", or "auto". CTranslate2 picks a compute type to match.
    device: str = "auto"
    #: "int8" on CPU is roughly four times quicker than "float32" for a quality
    #: cost that does not show up above `small`.
    compute_type: str = "default"
    #: Where checkpoints are cached. None uses the HuggingFace default.
    download_root: str | None = None
    #: Beam width. 1 is greedy and noticeably quicker; 5 is faster-whisper's
    #: default and better on hard audio.
    beam_size: int = 5
    info: ProviderInfo = None  # type: ignore[assignment]
    #: Real-time factor of the most recent transcription. The size priors are
    #: guesses about hardware koe cannot see; this is the measurement.
    last_rtf: float = field(default=0.0, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        spec = SIZES_BY_ID.get(self.size)
        if self.info is None:
            self.info = ProviderInfo(
                name="local-whisper",
                modality=Modality.ASR,
                model=self.size,
                languages=frozenset({Language.EN, Language.JA}),
                supports_word_timestamps=True,
                supports_language_hint=True,
                # Nothing is billed. What it costs is the RTF below.
                cost_per_audio_minute_usd=0.0,
                typical_rtf=spec.typical_rtf if spec else 1.0,
                expected_error_rate=dict(spec.error_rate) if spec else {},
            )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        """Load the checkpoint, fetching it first if it is not cached.

        Idempotent and safe to call concurrently: the lock is what stops two
        sessions starting at once from each downloading three gigabytes.
        """
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._build)

    def _build(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ProviderError(
                "local Whisper needs the 'asr' extra: pip install 'koe-harness[asr]'"
            ) from exc

        logger.info("loading whisper %s on %s", self.size, self.device)
        try:
            return WhisperModel(
                self.size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        except Exception as exc:
            raise ProviderError(f"could not load whisper {self.size!r}: {exc}") from exc

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: Language | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        await self.load()

        started = asyncio.get_running_loop().time()
        try:
            segments, detected = await asyncio.to_thread(self._run, audio, language, prompt)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"whisper failed: {exc}") from exc

        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
        # Logged rather than returned: the pipeline builds its own Usage from
        # the provider info, and the number worth having here is the *measured*
        # real-time factor, which is the one thing a size prior cannot know
        # about this particular machine.
        logger.debug(
            "whisper %s: %.1fs audio in %.0fms (rtf %.2f)",
            self.size,
            audio.duration,
            elapsed_ms,
            (elapsed_ms / 1000.0) / audio.duration if audio.duration else 0.0,
        )
        self.last_rtf = (elapsed_ms / 1000.0) / audio.duration if audio.duration else 0.0

        return Transcript(
            segments=segments,
            language=detected,
            duration=audio.duration,
            provider=self.info.name,
            model=self.size,
        )

    def _run(
        self, audio: AudioChunk, language: Language | None, prompt: str | None
    ) -> tuple[list[Segment], Language]:
        """The blocking part. Runs on a worker thread."""
        samples = _as_float32(audio)

        segments, info = self._model.transcribe(
            samples,
            language=_whisper_language(language),
            beam_size=self.beam_size,
            word_timestamps=True,
            # The user vocabulary rides in here where the provider takes one.
            # Biasing before the decode beats correcting after it.
            initial_prompt=prompt or None,
            # Whisper hallucinates fluent text over silence -- it is the
            # failure mode of the architecture, not a bug in a checkpoint. koe
            # already endpoints with its own VAD, so this is belt and braces
            # for the audio that does reach a decode.
            vad_filter=True,
            # faster-whisper's own hallucination detector, which is off by
            # default. It needs word timestamps, which koe asks for anyway, and
            # uses them to notice text attributed to a stretch of silence.
            hallucination_silence_threshold=2.0,
            # Off, and this is the important one. Feeding a window's output
            # back as the next window's context is what turns a single
            # hallucination into a loop: the model reads its own invention as
            # established fact and continues it. koe transcribes one utterance
            # per call, so the context this discards is worth little, and the
            # failure it prevents produces a paragraph of fluent nonsense.
            condition_on_previous_text=False,
            # A hard stop on the degenerate case, where the decoder falls into
            # repeating a phrase for the length of the buffer. Five is longer
            # than any phrase a person repeats verbatim and shorter than the
            # loops this prevents.
            no_repeat_ngram_size=5,
        )

        out: list[Segment] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            out.append(
                Segment(
                    text=text,
                    start=max(0.0, float(segment.start or 0.0)),
                    end=max(0.0, float(segment.end or 0.0)),
                    words=_words(segment),
                    language=_koe_language(getattr(info, "language", "")),
                    # avg_logprob is a log probability, not a confidence. It is
                    # mapped rather than passed through, or the number in the
                    # UI would be meaningless.
                    confidence=_confidence(segment),
                )
            )
        return out, _koe_language(getattr(info, "language", ""))


def _words(segment: Any) -> list[Word]:
    words = getattr(segment, "words", None) or []
    out: list[Word] = []
    for word in words:
        surface = (getattr(word, "word", "") or "").strip()
        if not surface:
            continue
        out.append(
            Word(
                text=surface,
                start=max(0.0, float(getattr(word, "start", 0.0) or 0.0)),
                end=max(0.0, float(getattr(word, "end", 0.0) or 0.0)),
                confidence=min(1.0, max(0.0, float(getattr(word, "probability", 1.0) or 1.0))),
            )
        )
    return out


def _confidence(segment: Any) -> float:
    """Map avg_logprob onto 0..1.

    Whisper reports a mean log probability per token, which is unbounded below
    and around -0.1 for confident output. exp() puts it on a probability scale;
    the clamp keeps a pathological segment from reporting a negative
    confidence that every downstream comparison would then get wrong.
    """
    import math

    raw = getattr(segment, "avg_logprob", None)
    if raw is None:
        return 1.0
    try:
        return min(1.0, max(0.0, math.exp(float(raw))))
    except (TypeError, ValueError, OverflowError):
        return 1.0


def _koe_language(code: str) -> Language:
    if code == "ja":
        return Language.JA
    if code == "en":
        return Language.EN
    return Language.UNKNOWN


def _as_float32(audio: AudioChunk) -> Any:
    """PCM16 bytes as the float32 array faster-whisper wants.

    numpy comes with the `asr` extra, so it is importable wherever
    faster-whisper is.
    """
    import array

    import numpy as np

    samples = array.array("h")
    samples.frombytes(audio.data[: len(audio.data) - (len(audio.data) % 2)])
    return np.asarray(samples, dtype=np.float32) / 32768.0
