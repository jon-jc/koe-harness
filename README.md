# koe（声）— a bilingual voice-AI harness

**Japanese/English voice AI, assembled.** Streaming ASR, speaker diarization and
an LLM writing 議事録 (meeting minutes), orchestrated behind one plugin kernel and
held to explicit **latency, cost and quality budgets**.

English · [日本語](README.ja.md)

[![CI](https://github.com/jon-jc/koe-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/jon-jc/koe-harness/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![TypeScript](https://img.shields.io/badge/client-TypeScript-3178c6)
![413 tests](https://img.shields.io/badge/tests-413-brightgreen)
![mypy strict](https://img.shields.io/badge/mypy-strict-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## The problem this is about

A voice product is never one model. It is an ASR model, a diarization model, an
LLM, and a lot of glue. The hard part is not any single model — it is the
**assembly**:

- Which backend should serve *this* request, given that live captions and an
  overnight archive job want opposite trade-offs?
- What happens when a vendor is slow, wrong, or down mid-meeting?
- How do you know a change made the product better rather than just different?
- What does an hour of audio actually cost, and which model is responsible?

koe answers those four questions as running code.

---

## What it does, concretely

```bash
pip install -e ".[dev,api,cli,ja]"
koe minutes            # transcript → 議事録, with verification shown
```

```
─── 1. transcript (ASR × diarization) ───
田中: 本日の議題は第三四半期の売上レビューです。
鈴木: 売上は前年比で百二十パーセント、目標を達成しました。
佐藤: はい、金曜日までに対応します。

─── 2. 議事録 ───
# 第三四半期 売上レビュー
**出席者**: 佐藤、田中、鈴木

## 決定事項
- 新機能のリリースは3月10日とする

## アクションアイテム
- [ ] KPIダッシュボードを更新する — 担当: 佐藤 / 期限: 金曜日

─── 3. verification ───
groundedness: 2/2 claims supported (100%)
repairs: 1
dropped 1 unsupported claim(s):
  - 来月中に全社展開を完了する        ← nobody said this. caught and removed.
```

That last line is the point of the whole LLM layer. See
[Hallucination detection](#hallucination-detection-that-is-a-string-operation).

---

## Architecture

```mermaid
flowchart LR
    subgraph Browser["Browser · TypeScript"]
        MIC[Microphone] --> AW[AudioWorklet<br/>48k float → 16k PCM16]
        AW -->|binary frames| WS
        WS -->|partial / final| UI[Live transcript<br/>committed vs pending]
    end

    subgraph Server["Server · Python"]
        WS[WebSocket] --> SESS[StreamingSession]
        SESS --> VAD[VAD + endpointing<br/>adaptive noise floor]
        VAD --> ROUTE{Router<br/>budget-aware}
        ROUTE -->|latency| ASR1[fast ASR]
        ROUTE -->|quality| ASR2[accurate ASR]
        ASR1 & ASR2 --> STAB[Stabilizer<br/>LocalAgreement]
        STAB --> FUSE[ASR × diarization<br/>fusion]
        FUSE --> MIN[議事録 generator]
        MIN --> GUARD[Groundedness<br/>guardrail]
    end

    subgraph Ops["Operations"]
        LEDGER[Cost ledger<br/>$/audio-hour]
        METRICS[EMF metrics<br/>p50/p95/p99]
        EVAL[Eval harness<br/>CER · DER · CIs]
    end

    ROUTE -.cost.-> LEDGER
    SESS -.latency.-> METRICS
    EVAL -.measured error rate.-> ROUTE
```

Right tool per layer, rather than one language everywhere:

| Layer | Stack | Why |
|---|---|---|
| Kernel, ASR/diarization, router, eval, LLM orchestration | **Python 3.11+** | Where the AI ecosystem, provider SDKs and eval statistics live |
| Realtime client — capture, resampling, streaming, live UI | **TypeScript** | `AudioWorklet` has no Python equivalent; browser audio belongs in the browser |
| Serving | **FastAPI + WebSocket** | Streaming-first, async throughout |
| Infra | **Docker + Terraform** (ECS Fargate) | Reproducible, boring, deployable |

---

## Four design positions

### 1. Japanese is a design constraint, not a locale string

This is the part most bilingual pipelines get wrong, and it does not fail
loudly — it produces numbers that look fine and mean nothing.

**WER is close to meaningless for Japanese.** The language has no word spaces,
so "words" only exist relative to a segmenter. Two teams reporting 12% WER on
the same audio with different tokenizers have not measured the same thing. koe
scores Japanese with **CER**, and every WER it reports carries the name of the
tokenizer that produced it.

**Normalization has two profiles, deliberately separated.** Scoring
normalization is aggressive and lossy; display normalization is conservative.
Conflate them and you either measure formatting instead of recognition, or strip
the 。and 、 that make Japanese readable:

```python
normalize_for_scoring("本日は、ＫＰＩを確認します。")  # → 本日はkpiを確認します
normalize_for_display("本日は、ＫＰＩを確認します。")  # → 本日は、KPIを確認します。
```

**Whitespace handling is language-aware.** Japanese has no word spaces, so any
the ASR inserted are arbitrary and must go — but English spaces are load-bearing.
A naive `.replace(" ", "")` turns `hello world` into `helloworld`:

```
その KPI を review します  →  そのkpiをreviewします
hello world               →  hello world
```

**漢数字 conversion is gated on morphology.** `一般` (general), `一緒`
(together) and `十分` (sufficient) all begin with numeral kanji and none are
numbers. Converting them yields `1般`, which is worse than doing nothing, and no
character-level rule can tell the difference — so conversion uses MeCab POS tags
and falls back to a conservative stoplist when MeCab is absent.

**Endpointing is tuned per language.** Japanese speakers pause before
sentence-final particles and politeness endings (〜ですね, 〜ますので). An
English-tuned endpointer cuts there and removes the verb — which in Japanese is
where negation and tense live. JA gets 900 ms of silence tolerance, EN gets 650.

### 2. Model choice is a runtime decision

Backends register as services carrying their cost, speed and expected quality as
data. A request declares a budget; the router filters on hard constraints, then
ranks what survives:

```python
Budget.realtime()   # live captions: <800ms, streaming required
Budget.accurate()   # post-meeting 議事録: no latency cap, best model
Budget.bulk()       # archive re-processing: cost priority, quality floor
```

Hard constraints **eliminate** rather than penalize — a 200 ms cap on live
captions is not a preference a very cheap backend can outweigh by being cheap.
Scores are normalized *within the candidate set*, so when every candidate costs
about the same, cost stops influencing the ranking and quality decides.

**The loop closes with evaluation.** `record_measurement()` replaces a
provider's documented prior with a measured error rate, so routing converges on
reality instead of vendor marketing:

```python
router.select(budget).chosen_name                          # "vendor_a" (prior)
router.record_measurement("vendor_a", Language.JA, 0.22)   # production says otherwise
router.select(budget).chosen_name                          # "vendor_b"
```

### 3. Uncertainty is quantified, not asserted

An error rate on forty utterances is an estimate, not a measurement. Every
number carries a bootstrap confidence interval, and A/B comparisons get a paired
bootstrap plus a paired permutation test:

```
$ koe compare --baseline 0.0 --candidate 0.12
delta            +10.24%   CI [+8.48%, +12.11%]   p=0.0005  →  REGRESSION

$ koe compare --baseline 0.10 --candidate 0.105
delta            -0.61%    CI [-3.02%, +1.74%]    p=0.6742  →  INCONCLUSIVE
```

Resampling happens at the **utterance** level, never the character level —
errors within an utterance are strongly correlated, and treating characters as
independent would understate variance badly and produce intervals far too narrow
to be honest.

**Regression gates fail only on significant regressions.** A gate that fires on
run-to-run noise gets re-run until green and then ignored, so a difference that
cannot be distinguished from noise is a warning, not a build failure.

**Slicing is where the findings are.** A corpus-level CER averages away the
thing you can act on:

| slice | n | CER |
|---|---|---|
| overall | 41 | 8.40% |
| tag:monolingual | 32 | 7.83% |
| **tag:code-switch** | 9 | **10.59%** |
| lang:ja | 28 | 10.20% |
| lang:en | 13 | 6.42% |

### 4. Hallucination detection that is a string operation

<a name="hallucination-detection-that-is-a-string-operation"></a>

An LLM asked to summarize a meeting will occasionally invent an action item
nobody agreed to, or attach a deadline nobody said — fluently enough that a
reviewer skims past it. In a minutes product, that invented task lands in
someone's backlog with a due date.

So the schema **requires a verbatim `source_quote` on every claim**. That single
constraint turns hallucination detection into a string operation: either the
quote is in the transcript or it isn't. No second model call, no LLM-judge
uncertainty of its own, effectively zero cost.

Matching runs on scoring-normalized text (reusing the layer above), so a model
that copies verbatim but differs in width or punctuation still matches. Support
is scored by longest common substring rather than pass/fail, so a lightly
reworded quote scores ~0.9 and a fabricated one scores near 0 — which lets the
system keep the first and drop the second.

Unsupported claims are **dropped, not flagged**: an action item marked
"unverified" still ends up in a task list, and the reader has no way to check it.

---

## Reliability

| Guarantee | Why it exists |
|---|---|
| Teardown always completes | One failing cleanup never strands its siblings — a leaked socket in one plugin must not leak the rest |
| Children die before parents | A decoder cannot outlive the session owning its audio buffer |
| A crashing subscriber cannot drop audio | Handler failures are isolated per emit |
| Committed captions are never revised | Taking back text a viewer already read is worse than having waited |
| Audio intake never blocks on recognition | Dropping a caller's speech to wait on a slow model is unrecoverable |
| A full server rejects, never queues | A caller queued behind a full server records audio nobody is transcribing |
| Non-retryable errors stop the fallback chain | A malformed request fails identically at the next provider |
| Only retryable failures open a circuit breaker | A bad request says nothing about provider health |
| Production refuses to boot misconfigured | A wildcard CORS policy should stop a deploy, not become an incident |

---

## Quick start

Everything runs on deterministic mocks — **no API keys required**.

```bash
python -m venv .venv && source .venv/bin/activate   # .venv/Scripts/activate on Windows
pip install -e ".[dev,api,cli,ja]"
pytest                                              # 400 tests
```

```bash
koe info                    # which backends are available here
koe minutes                 # 議事録 generation, with the guardrail catching a fabrication
koe evaluate                # score a system, sliced by language and tag
koe compare --baseline 0.0 --candidate 0.12   # A/B with a significance verdict
koe gate                    # CI regression gates; exits non-zero on failure
koe serve                   # API + live web client on :8000
```

Real providers activate when credentials are present:

```bash
export KOE_ANTHROPIC_API_KEY=sk-...
koe minutes --live
```

### The live client

```bash
cd web && npm install && npm run build && cd ..
koe serve      # → http://127.0.0.1:8000
```

Microphone audio is downsampled to 16 kHz mono PCM16 in an `AudioWorklet` before
it leaves the browser — ~6× less uplink than 48 kHz float, and off the main
thread so a UI repaint cannot drop input frames. Committed text renders normally
while the tail the model has not settled on is greyed.

### Desktop application

```bash
pip install -e ".[api,cli,ja,desktop]"
python -m koe.desktop            # or: koe-desktop
```

A native window over the same API the server deployment runs — nothing is
stubbed for desktop, so the two builds cannot drift apart. Build an installer
with `python packaging/build.py --installer`: a 47 MB `koe-setup-<version>.exe`
that installs per-user with no UAC prompt.

pywebview over WebView2 rather than Electron: the client already exists and is
38 KB, and Windows already has the browser engine — shipping a second copy of
Chromium to run it would add ~150 MB for nothing. See
[docs/desktop.md](docs/desktop.md) for single-instance handling, crash
reporting, the capability probe, and what is deliberately not done (the binary
is unsigned).

### Docker

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
```

Multi-stage build, non-root (uid 10001), read-only root filesystem, healthcheck.
Verified locally: **healthy in 2 s with no credentials**, MeCab present, real
WebSocket session produces a transcript.

---

## Project layout

```
src/koe/
  kernel/        Plugin core — scopes, reactive services, event bus
  text/          JA/EN script analysis, normalization, 漢数字, tokenization
  domain/        Audio (hot path, dataclasses) and transcripts (pydantic)
  providers/     ASR / diarization / LLM behind narrow protocols
    llm/         Anthropic + OpenAI adapters
  routing/       Budgets, circuit breakers, fallback chains
  evaluation/    CER/WER/DER, bootstrap CIs, regression gates, corpus
  pipeline/      VAD, endpointing, stabilization, streaming sessions
  minutes/       議事録 schema, prompts, guardrails, generator
  telemetry/     Cost ledger, metrics, structured logging
  api/           FastAPI + WebSocket
  cli/           koe command line
web/src/         TypeScript realtime client
deploy/          Dockerfile, compose, Terraform (ECS Fargate)
```

---

## Testing and CI

400 tests, `mypy --strict` clean, `ruff` clean, `tsc --noEmit` clean.

CI runs eight jobs on every push:

- **lint & types** — ruff lint + format, mypy strict
- **tests** × 4 — Python 3.11/3.12 × with and without MeCab
- **web client** — `tsc --noEmit`, build, and a check that the committed bundle is not stale
- **terraform** — `fmt -check` and `validate`
- **docker** — builds the image and asserts it becomes healthy with no credentials

The MeCab matrix split is deliberate: Japanese tokenization is an optional
dependency, which creates a hazard specific to this project — a dev machine with
MeCab and a runner without it can silently disagree, so a normalization bug
reproduces in one place and not the other. `tests/text/test_backend_parity.py`
pins the contract by forcing each backend explicitly.

CI also smoke-tests the evaluation harness itself: a clean backend must score 0,
and a degraded one must be caught as a significant regression. The harness that
guards quality needs guarding too.

---

## Documentation

| Document | |
|---|---|
| [Architecture](docs/architecture.md) | The decisions where an alternative was seriously considered and rejected, with the reason |
| [Privacy & data handling](SAFETY.md) · [日本語](SAFETY.ja.md) | What touches audio, what is retained, third-party exposure, APPI |
| [Benchmarks](BENCHMARK.md) | How performance is measured, and the numbers |
| [Desktop application](docs/desktop.md) | Packaging, the .exe, and desktop-specific failure modes |
| [Contributing](CONTRIBUTING.md) | Setup, conventions, and what reviewers look for |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dependencies and prior art |

---

## Deliberate limitations

Being explicit about what this is and isn't:

- **The default backends are deterministic mocks.** Real adapters exist for
  Anthropic, OpenAI and (via protocol) Whisper/pyannote, but the mocks are what
  CI and the demo run on. That is a feature — it keeps CI free, deterministic,
  and runnable on a clean checkout — but the reported error rates measure the
  *harness*, not any real ASR model.
- **The corpus is 41 synthetic utterances**, hand-written to contain the cases
  that break Japanese pipelines. It validates the machinery; it is not a
  benchmark result. Swapping in real recordings changes the data, not the code.
- **Sessions live in process.** The ALB is configured sticky and scaling is
  horizontal per-task. Cross-instance session migration would need a shared
  store, which this does not have.
- **The speaker mapping in DER is greedy**, not Hungarian. It is exact whenever
  one hypothesis label dominates each reference speaker — the normal case for
  meetings — and near-optimal otherwise.
- **Terraform is validated, not applied.** `fmt` and `validate` run in CI; it has
  not been deployed to a live AWS account.

## Acknowledgements

The plugin kernel's composability model — scoped lifetimes, dependency-gated
activation, spatial filters — is adapted from
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) and the
[Cordis](https://cordis.js.org) "everything is a plugin" design, reimplemented in
Python and specialized for a realtime audio domain.

Partial stabilization uses **LocalAgreement**, as described by Macháček et al.
and used in `whisper_streaming`.

## License

MIT
