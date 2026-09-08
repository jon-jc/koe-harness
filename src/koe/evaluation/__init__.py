"""Evaluation: metrics, uncertainty, and regression gates.

The harness answers three questions that are usually conflated:

* **How good is this system?** -- :mod:`koe.evaluation.metrics`, with CER as the
  primary Japanese metric and every WER carrying its tokenizer's name.
* **How sure are we?** -- :mod:`koe.evaluation.statistics`, bootstrap intervals
  and paired permutation tests, because an error rate on forty utterances is an
  estimate rather than a measurement.
* **Did this change make it worse?** -- :mod:`koe.evaluation.regression`, which
  blocks a merge only on regressions that are statistically significant, so the
  gate stays credible enough to leave switched on.
"""

from koe.evaluation.dataset import Dataset, EvalCase, ScriptLine
from koe.evaluation.metrics import (
    AlignmentOp,
    DiarizationScore,
    ErrorRate,
    TranscriptionScore,
    Unit,
    align,
    character_error_rate,
    diarization_error_rate,
    score_transcription,
    speaker_count_error,
    word_error_rate,
)
from koe.evaluation.regression import (
    DEFAULT_GATES,
    Gate,
    GateResult,
    Metric,
    RegressionReport,
    check_regression,
    paired_samples,
)
from koe.evaluation.runner import (
    CaseResult,
    EvalReport,
    Slice,
    mock_transcriber,
    run_evaluation,
)
from koe.evaluation.statistics import (
    Comparison,
    ConfidenceInterval,
    MetricSummary,
    bootstrap_interval,
    paired_bootstrap,
    paired_permutation_test,
    summarize,
)

__all__ = [
    "DEFAULT_GATES",
    "AlignmentOp",
    "CaseResult",
    "Comparison",
    "ConfidenceInterval",
    "Dataset",
    "DiarizationScore",
    "ErrorRate",
    "EvalCase",
    "EvalReport",
    "Gate",
    "GateResult",
    "Metric",
    "MetricSummary",
    "RegressionReport",
    "ScriptLine",
    "Slice",
    "TranscriptionScore",
    "Unit",
    "align",
    "bootstrap_interval",
    "character_error_rate",
    "check_regression",
    "diarization_error_rate",
    "mock_transcriber",
    "paired_bootstrap",
    "paired_permutation_test",
    "paired_samples",
    "run_evaluation",
    "score_transcription",
    "speaker_count_error",
    "summarize",
    "word_error_rate",
]
