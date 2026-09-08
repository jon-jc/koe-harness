# Benchmarks

```bash
python -m benchmarks               # all groups
python -m benchmarks --only audio  # one group
koe bench --markdown               # emit the tables below
```

## What is measured, and why these things

Only components whose cost actually constrains the product:

- **Text processing** runs on every utterance — twice during evaluation (once
  for the reference, once for the hypothesis) and once on the realtime path.
- **Metrics** run over the whole corpus on every eval, and the bootstrap layers
  2,000 resamples on top. It is the slowest thing in CI by a wide margin.
- **The audio path** is where real-time factor decides whether concurrency is
  possible at all.
- **Routing** happens per request, so it has to be cheap enough to be free.

Nothing here measures a model. The ASR and LLM backends are deterministic
mocks, so these numbers describe **the harness overhead around a model**, not
inference. That is the useful thing to know: it tells you how much of a latency
budget is left for the model, which is the only part you cannot optimize by
writing better Python.

## Method

- **Warmup** before every measurement. The first call pays for imports and the
  MeCab dictionary load; timing it reports startup cost as steady-state
  throughput.
- **Percentiles, not means** — same reason the metrics layer reports them. A
  mean is dominated by the fast majority and hides the tail, and on a realtime
  path the tail is what breaks a budget.
- **GC disabled during the timed loop**, collected once before it. Collection
  during a run shows up as a tail belonging to the allocator rather than to the
  code under test.
- **Real-time factor** for anything on the audio path. "0.5 ms per call" says
  nothing without knowing how much audio the call covered; RTF — processing
  seconds per audio second — is what says whether a stream keeps up.

## Results

Measured on Python 3.12.10, Windows AMD64, MeCab available. **Numbers from one
developer machine — treat the ratios as meaningful and the absolutes as
indicative.** CI does not gate on them.

### text

| benchmark | p50 | p95 | throughput |
|---|---:|---:|---:|
| normalize (scoring, JA) | 0.112 ms | 0.123 ms | 313,901 char/s |
| normalize (display, JA) | 0.032 ms | 0.035 ms | 1,083,591 char/s |
| normalize (scoring, EN) | 0.138 ms | 0.150 ms | 558,376 char/s |
| normalize (scoring, mixed) | 0.100 ms | 0.106 ms | 460,922 char/s |
| normalize (scoring, 1.4k chars) | 2.030 ms | 2.355 ms | 344,853 char/s |
| detect_language (mixed) | 0.030 ms | 0.035 ms | 1,523,179 char/s |
| 漢数字 → arabic | 0.015 ms | 0.017 ms | 2,258,065 char/s |
| tokenize JA | 0.043 ms | 0.050 ms | 804,598 char/s |

### metrics

| benchmark | p50 | p95 | throughput |
|---|---:|---:|---:|
| CER (one utterance) | 0.385 ms | 0.460 ms | 90,933 char/s |
| WER (one utterance) | 0.372 ms | 0.411 ms | 206,989 char/s |
| CER (41-case corpus) | 10.561 ms | 10.855 ms | 3,882 case/s |
| DER (60 s, 3 speakers) | 1.005 ms | 1.074 ms | 995 op/s |
| bootstrap CI (2,000 resamples) | 15.027 ms | 15.634 ms | 67 op/s |
| paired bootstrap + permutation | 16.319 ms | 16.607 ms | 61 op/s |

### audio path

| benchmark | p50 | p95 | RTF |
|---|---:|---:|---:|
| VAD (1 s of audio) | 0.487 ms | 0.544 ms | `4.9e-04` |
| stabilizer update | 0.022 ms | 0.025 ms | — |
| ASR × diarization fusion (24 s meeting) | 0.014 ms | 0.017 ms | `6.0e-07` |
| mock ASR transcribe (4.3 s utterance) | 0.334 ms | 0.407 ms | `7.8e-05` |

### routing & guardrails

| benchmark | p50 | p95 | throughput |
|---|---:|---:|---:|
| router.select (5 providers) | 0.005 ms | 0.005 ms | 212,766 op/s |
| groundedness check (4 claims) | 2.093 ms | 2.490 ms | 478 op/s |

## What the numbers say

**The harness is not the bottleneck, by three orders of magnitude.** The
slowest audio-path stage is VAD at RTF 4.9×10⁻⁴ — roughly 2,000 concurrent
streams per core as a ceiling. That ceiling ignores GIL contention and network
waits and is not a capacity plan, but it establishes the useful fact: when a
session is slow, the model is slow, and the orchestration around it is
rounding error.

**Scoring normalization costs 3.5× display normalization** (0.112 ms vs
0.032 ms), and the entire difference is morphology-gated numeral conversion.
That is the price of not turning `一般` into `1般`, paid once per utterance per
side, and it is worth it — but it is also why the two profiles are separate.
The realtime path uses the cheap one.

**The bootstrap dominates evaluation.** A 41-case corpus scores in 10.6 ms;
putting a confidence interval on that result costs 15 ms, and a paired
comparison 16 ms — more than the measurement itself. This is the right trade
(an error rate without an interval is not a result), and it is why iteration
count is a parameter rather than a constant.

**Routing is free.** 0.005 ms per selection means the router can run per
request without anyone needing to think about it, which is what makes
per-request budgets practical rather than aspirational.

**Groundedness checking is the most expensive non-statistical operation** at
2.1 ms for four claims, from `difflib.SequenceMatcher` doing longest-common-
substring against the transcript. It is still four orders of magnitude cheaper
than the LLM call that produced the claims, which is the entire argument for
citation-based verification over an LLM judge.

## Interpreting these honestly

- **Mocks, not models.** These measure harness overhead. A real Whisper pass
  runs at RTF 0.1–0.3 on GPU and above 1.0 on many CPUs — several thousand
  times the numbers here. The harness is designed to disappear into that.
- **One machine, one run.** No cross-machine comparison, no CI gating, no
  regression detection on performance. The quality regression gates
  (`koe gate`) exist; a performance equivalent does not.
- **Synthetic inputs.** Real audio has noise, overlapping speech and
  disfluencies that change VAD and ASR behaviour. It would not change the
  ratios here, since none of these components branch on content.
- **Single-threaded.** Everything is measured in one process on one core with
  no concurrent load. The `max_concurrent` figure is arithmetic on RTF, not a
  load test.

A real performance regression gate would need a stable runner, a warm cache,
and a statistical treatment of run-to-run variance — the same machinery the
quality gates already have. That is on the list in
[docs/architecture.md](docs/architecture.md), not in this repository yet.
