# koe (声) — a bilingual voice-AI harness

**日本語 / English.** A plugin harness for building voice products out of several
AI models at once — streaming ASR, speaker diarization, and an LLM writing
議事録 (meeting minutes) — and holding the whole assembly to explicit **latency,
cost, and quality budgets**.

> 複数の AI モデル（音声認識・話者分離・LLM）を組み合わせて音声プロダクトを構築するための
> プラグイン・ハーネス。レイテンシ・コスト・品質のトレードオフを明示的に管理します。

[![CI](https://github.com/jon-jc/koe-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/jon-jc/koe-harness/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

A voice product is never one model. It is an ASR model, a diarization model, an
LLM, and a pile of glue — and the hard part is not any single model, it is the
**assembly**: which model to call, what to do when one is slow or wrong, how to
know whether a change made the product better or just different, and what the
whole thing costs per hour of audio.

koe is that assembly layer, built around three positions:

**1. Model choice is a runtime decision, not a deployment.**
ASR backends are registered as services. Swapping one rebuilds exactly the
plugins that depend on it and leaves everything else running — no redeploy, no
dropped sessions. The router picks a backend per request from a declared
latency/cost/quality budget rather than a hardcoded default.

**2. Japanese is a first-class design constraint, not a locale string.**
Word Error Rate is close to meaningless for Japanese: the language has no
spaces, so "words" depend on which tokenizer you picked. koe scores Japanese
with **CER** and MeCab-tokenized WER, normalizes 全角・半角, NFKC forms, and
漢数字 (一万二千 ↔ 12000) before comparison, and handles the JA↔EN
code-switching that is normal in Japanese business meetings. Getting this wrong
does not produce a slightly worse number — it produces a number that means
nothing.

**3. LLM output is uncertain, so the system is built to measure it.**
Every summarization result is checked for groundedness against the transcript,
validated against a schema, and repaired-or-rejected on failure. The eval
harness reports **bootstrap confidence intervals on metric deltas**, so
"this prompt is better" is a claim with an interval attached rather than a vibe.

---

## Architecture

Right tool per layer, rather than one language everywhere:

| Layer | Stack | Why |
|---|---|---|
| Harness kernel, ASR/diarization, router, eval, LLM orchestration | **Python 3.11+** | Where the AI ecosystem, the provider SDKs, and the eval statistics live |
| Realtime voice client — mic capture, PCM downsampling, WS streaming, live transcript | **TypeScript** | `AudioWorklet` has no Python equivalent; browser audio belongs in the browser |
| Serving | FastAPI + WebSocket | Streaming-first, async all the way down |
| Infra | Docker + Terraform (ECS Fargate) | Reproducible, boring, deployable |

### The kernel

The plugin core is modelled on [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
and the [Cordis](https://cordis.js.org) "everything is a plugin" design, ported
to Python and specialized for a realtime audio domain. The idea worth stealing
is that a plugin's lifetime is composable along **two axes**:

**Temporal — a plugin lives exactly as long as its dependencies.**

```python
@plugin(name="minutes", inject=["llm", "transcript"])
def minutes_plugin(ctx, config):
    ctx.on("session.end", write_minutes)

fork = ctx.plugin(minutes_plugin)   # dormant: nothing to depend on yet
ctx.provide("llm", claude)          # still dormant, transcript missing
ctx.provide("transcript", store)    # -> starts here
ctx.provide("llm", gpt, replace=True)  # -> torn down and rebuilt on the new client
```

Nothing polls, and no plugin writes a reconnect path.

**Spatial — a plugin can be live for only part of the traffic.**

```python
ja = ctx.select(lambda utt: utt.language == "ja")
ja.plugin(minutes_plugin, MinutesConfig(style="keigo"))
```

That plugin is, from its own point of view, completely ordinary. It simply
never sees English audio.

The two compose, which is the whole point: *"summarize in keigo, for Japanese
sessions only, whenever an LLM is available"* is a declaration, not a branch
buried in a handler.

### Guarantees the kernel makes

- **Teardown always completes.** One failing cleanup callback never strands its
  siblings; failures are aggregated and raised together. A leaked socket in one
  plugin must not leak the rest.
- **Children die before parents.** A decoder cannot outlive the session owning
  its audio buffer.
- **A crashing subscriber cannot drop audio.** Handler failures are isolated
  per-emit.
- **Slow handlers are visible.** Anything over budget is logged with its event
  name, because on a realtime path a slow subscriber is a latency bug that
  would otherwise hide.

---

## Status

Built in milestones, each merged via its own PR.

- [x] **M1 — Kernel.** Scopes, reactive services, event bus, plugin lifecycle.
- [x] **M2 — Bilingual text layer.** Script analysis, JA/EN language ID with code-switch detection, 漢数字 parsing, scoring/display normalization profiles, MeCab tokenization with a labelled fallback.
- [ ] **M3 — Domain & providers.** Typed audio/transcript models, ASR + diarization + LLM adapters.
- [ ] **M4 — Evaluation.** CER/WER/DER, bootstrap CIs, regression gate.
- [ ] **M5 — Routing.** Budget-aware model selection, circuit breakers, fallback chains.
- [ ] **M6 — Realtime pipeline.** VAD, endpointing, partial stabilization, ASR×diarization fusion.
- [ ] **M7 — LLM layer.** 議事録 generation, groundedness guardrails, schema repair.
- [ ] **M8 — Serving.** FastAPI + WebSocket, TypeScript voice client.
- [ ] **M9 — Operations.** Cost ledger, metrics, Docker, Terraform.

---

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -e ".[dev]"
pytest
```

## License

MIT
