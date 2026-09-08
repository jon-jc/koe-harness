# Contributing

## Getting set up

```bash
python -m venv .venv && . .venv/Scripts/activate   # .venv/bin/activate on POSIX
pip install -e ".[ci]"                              # dev + api + cli + ja
pytest
```

The `ci` extra is deliberately the one to install: it is the same dependency
set CI uses, so "passes locally" means something. Installing only `dev` will
skip the Japanese tokenizer and the API tests, and you will find that out from
a red build rather than from your terminal.

For the web client:

```bash
cd web && npm install && npm run build
```

## Before opening a pull request

```bash
ruff format src tests
ruff check src tests
mypy
pytest
cd web && npx tsc --noEmit && npm run build
```

All five must pass. CI runs the same commands plus a Terraform validate and a
Docker build, across Python 3.11/3.12 with and without MeCab.

**Commit the web bundle.** `web/dist/app.js` and `web/dist/app.css` are checked
in so `koe serve` works without a Node toolchain. CI rebuilds them and fails if
the committed copies are stale.

## What good looks like here

This codebase has a consistent style that is more about *reasoning* than
formatting. A few conventions that reviewers will look for:

### Comments explain why, not what

The code says what it does. A comment earns its place by saying why it is that
way, and ideally what breaks otherwise.

```python
# Bad — restates the code
# Adapt the noise floor
self._noise_floor_db = (1 - alpha) * self._noise_floor_db + alpha * energy_db

# Good — says why, and what fails without it
# Adapt only on non-speech, or the floor climbs to meet the speaker and the
# detector goes deaf partway through a sentence.
```

### Tests assert behaviour with a stated reason

Test names read as claims; the docstring says why the claim matters.

```python
def test_a_word_far_from_every_turn_is_unknown_rather_than_guessed() -> None:
    """Confidently attributing a commitment to the wrong person is the worst outcome."""
```

### Japanese is a design constraint, not a locale

If you touch text handling, evaluation, or the UI, check the Japanese path
specifically:

- Is this scoring or display? The two normalization profiles are not
  interchangeable (`normalize_for_scoring` vs `normalize_for_display`).
- Does it assume word boundaries? Japanese has none. WER without a named
  tokenizer is not a comparable number.
- Does it assume whitespace is insignificant, or significant? It is one or the
  other depending on the script on either side of it.
- Does it work with and without MeCab? `tests/text/test_backend_parity.py`
  exists because a dev machine and a CI runner disagreeing on this is a real
  hazard.

### Failure modes are decided, not defaulted

Where something can fail, the code should say what happens and why that is the
right answer — a dropped frame, a rejected connection, a claim removed from a
document. "It throws" is rarely the considered choice on a realtime path.

## Adding a provider

Implement the protocol in `koe/providers/base.py` and publish a `ProviderInfo`
with honest cost, latency and quality numbers:

```python
info = ProviderInfo(
    name="my-asr",
    modality=Modality.ASR,
    model="my-model-v1",
    languages=frozenset({Language.JA, Language.EN}),
    cost_per_audio_minute_usd=0.004,
    typical_first_result_ms=250.0,
    expected_error_rate={Language.JA: 0.09},   # a documented prior
)
```

`expected_error_rate` starts as a prior and is replaced by measurement from the
eval harness via `Router.record_measurement()`. **Do not put a flattering
number there.** An unmeasured provider reports `1.0` by design — unknown
quality is treated as bad rather than as free — so a fabricated prior mainly
buys you routing decisions you did not intend.

## Adding evaluation cases

Cases live in `koe/evaluation/corpus.py` and are **utterances, not meetings**.
That is not an aesthetic preference: the bootstrap resamples cases, so the case
count is the sample size, and a paired permutation test on three cases cannot
produce a p-value below ~0.25 regardless of effect size.

Tags are derived from content in `derive_tags()` rather than hand-assigned, so
they cannot rot as the corpus grows. If you add a property worth slicing on,
add it there.

## Project layout

```
src/koe/
  kernel/        Plugin core — scopes, reactive services, event bus
  text/          JA/EN script analysis, normalization, 漢数字, tokenization
  domain/        Audio (hot path) and transcripts (API boundary)
  providers/     ASR / diarization / LLM behind narrow protocols
  routing/       Budgets, circuit breakers, fallback chains
  evaluation/    CER/WER/DER, bootstrap CIs, regression gates, corpus
  pipeline/      VAD, endpointing, stabilization, streaming sessions
  minutes/       議事録 schema, prompts, guardrails, generator
  telemetry/     Cost ledger, metrics, structured logging
  api/           FastAPI + WebSocket
  cli/           koe command line
web/src/         TypeScript realtime client
deploy/          Dockerfile, compose, Terraform
```

Dependencies point one way, top to bottom. `kernel` knows nothing about audio;
`text` knows nothing about models; `domain` knows nothing about providers.
`evaluation` and `routing` are joined by exactly one call
(`record_measurement`), because a cycle between "how good is this model" and
"which model should we use" makes both untestable in isolation.

## Documentation

`docs/architecture.md` records decisions where an alternative was seriously
considered and rejected, with the reason. If you make a choice a future reader
would otherwise have to reverse-engineer, add it there.

Japanese translations (`README.ja.md`, `SAFETY.ja.md`) are written, not
machine-translated. If you change the English and cannot update the Japanese,
say so in the PR rather than leaving them silently divergent.
