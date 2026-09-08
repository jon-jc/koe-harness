# Architecture

This document covers the decisions behind koe that are not obvious from reading
any single module — the ones where an alternative was seriously considered and
rejected for a stated reason.

The README covers what the system does. This covers why it is shaped this way.

---

## 1. Layering

```
kernel      composition: scopes, services, events        (knows nothing about audio)
   ↓
text        JA/EN script analysis, normalization         (knows nothing about models)
   ↓
domain      audio and transcript types                   (knows nothing about providers)
   ↓
providers   ASR / diarization / LLM behind protocols     (knows nothing about routing)
   ↓
routing     budget-aware selection, breakers, fallback   (knows nothing about sessions)
   ↓
pipeline    VAD, stabilization, streaming sessions       (knows nothing about HTTP)
   ↓
api         FastAPI + WebSocket
```

Dependencies point one way. The two layers worth defending:

**`text` sits below `domain`.** Normalization is not a presentation concern here
— it is what makes two transcripts comparable, so the evaluation layer, the
guardrails and the stabilizer all depend on it. Putting it near the top, where
"text formatting" usually goes, would force three separate layers to reach
upward.

**`evaluation` and `routing` do not depend on each other.** They are joined by a
single explicit call (`router.record_measurement(...)`), because a circular
dependency between "how good is this model" and "which model should we use" is
how a system becomes impossible to test in pieces.

---

## 2. Why a plugin kernel at all

The obvious alternative is a straight function pipeline:
`audio → vad → asr → diarize → fuse → summarize`. It is simpler, and for a
single-model product it would be the right answer.

It stops working when the product needs behaviour that varies by *slice of
traffic* and by *runtime availability* at the same time. Concretely: "summarize
in keigo, for Japanese sessions only, whenever an LLM is available." In a
straight pipeline that becomes a conditional inside the summarize step, then
another one inside the persist step, then a third in the metrics step — and the
knowledge that these three conditionals are the same policy exists only in
someone's head.

The kernel makes it one declaration:

```python
ja = ctx.select(lambda utt: utt.language == "ja")
ja.plugin(minutes_plugin, MinutesConfig(style="keigo"))   # inject=["llm"]
```

Two composable axes:

- **Temporal** — a plugin lives exactly as long as the services it declares.
  Registering an ASR backend starts every ASR-dependent plugin; replacing one
  tears down and rebuilds *only its dependents*. Nothing polls, and no plugin
  writes a reconnect path.
- **Spatial** — `ctx.select(pred)` narrows a context so its subscriptions fire
  only for matching payloads. The plugin is, from its own point of view, an
  ordinary plugin that simply never sees English audio.

### The cost of this choice

It is more machinery than a small product needs, and the indirection makes a
stack trace longer. The trade is deliberate: the alternative failure mode —
policy scattered across conditionals in unrelated modules — is the one that
compounds, and it is much harder to remove later than a kernel is.

### Scope discipline

Every plugin's effects are collected into a scope. Disposal:

- runs children before parents (a decoder cannot outlive its session's buffer)
- runs callbacks in reverse registration order
- **always completes** — a failing callback is collected, not allowed to abort
  the rest, and failures are aggregated into one `DisposalError`

That last rule exists because the realtime path builds a scope per call leg. A
partial teardown is a socket leak *per dropped call*, which is the shape of leak
that only shows up under the load you cannot reproduce.

---

## 3. Why Japanese changes the design rather than configuring it

Four English-shaped assumptions break, and each one breaks *quietly*.

### Word Error Rate assumes words exist

Japanese has no word spaces. "Words" are whatever a segmenter says they are, so
a Japanese WER is only meaningful alongside the name of the tokenizer that
produced it. Two teams reporting 12% on the same audio with different segmenters
have not measured the same thing.

koe reports **CER as primary for Japanese**, and attaches the tokenizer name to
every WER it produces. When MeCab is unavailable it falls back to character
segmentation and *says so* — a labelled fallback rather than a silent one,
because a number that looks comparable and isn't is worse than a missing number.

### Whitespace is arbitrary on one side and load-bearing on the other

Japanese ASR output contains spaces the model invented. English spaces separate
words. A single rule cannot serve both, so the normalizer deletes a space only
when Japanese script sits on one side of it:

```
その KPI を review します  →  そのkpiをreviewします
hello world               →  hello world
```

### Numbers have three written forms

`2025年`, `二〇二五年` and `二千二十五年` are the same content. Scoring them
against each other without folding measures formatting, not recognition.

The hard part is knowing what is a number at all: `一般`, `一緒` and `十分` all
begin with numeral kanji and none are numbers. `一般 → 1般` is worse than doing
nothing, and no character-level rule can distinguish them — it needs morphology.
So conversion is gated on MeCab POS tags (convert only 名詞-数詞), with a
conservative stoplist as the fallback. **The fallback deliberately
under-converts.**

A bug found by this design: MeCab segments `二千二十五` into `二千`/`二十`/`五`,
and converting each token independently concatenates to `2000205`. Adjacent
numeral tokens must be folded into one run before parsing.

### Silence means something different

Japanese speakers pause before sentence-final particles and politeness endings
(〜ですね, 〜ますので). An English-tuned endpointer reads that pause as "done" and
cuts — removing the verb, which in Japanese carries the negation and the tense.
JA gets 900 ms of tolerance, EN 650 ms.

---

## 4. Two normalization profiles

The single most useful distinction in the text layer:

| | Scoring | Display |
|---|---|---|
| Purpose | make two transcripts comparable | make text correct for a reader |
| Punctuation | stripped | kept |
| Case | folded | kept |
| JA whitespace | removed | kept |
| Numerals | folded to Arabic | untouched |
| Fillers | removed | kept |

Applied identically to reference and hypothesis, the scoring profile removes
formatting from the error rate. Skip it and a model writing `2025年` scores worse
than an identical model writing `二〇二五年`, which tells you nothing about
either. Run it on user-visible output and you strip the 。and 、 that make
Japanese readable.

One subtlety: intra-word apostrophes survive punctuation stripping, because
removing the apostrophe from `we're` yields `were` — a different word — so the
"harmless" strip silently invents a substitution error.

---

## 5. Statistical honesty in evaluation

An error rate computed on a small eval set is an estimate. Reporting "CER
improved from 8.1% to 7.6%" without saying whether that could be noise is how
teams ship regressions believing they shipped improvements.

**Bootstrap intervals** for "how good, give or take". **Paired bootstrap plus
paired permutation** for "is B actually better".

Three decisions:

1. **Resampling is at the utterance level, never the character level.** Errors
   within an utterance are strongly correlated — a model that loses the audio
   mid-sentence gets the whole clause wrong. Treating characters as independent
   would understate variance by a large factor and produce intervals far too
   narrow to be honest.
2. **Comparisons are paired.** Both systems see the same utterances, so
   resampling the same indices for both cancels between-utterance difficulty and
   leaves only the difference between systems. This routinely turns an
   inconclusive comparison into a decisive one without touching the data.
3. **Corpus rates are pooled, not averaged.** A mean of per-utterance rates
   weights a two-word utterance the same as a two-hundred-word one, so it moves
   when segmentation changes even though the audio did not.

### A finding that changed the corpus design

The corpus began as 3 meeting-level cases. With n=3 a paired permutation test has
only 2³ = 8 possible sign flips, so a two-sided p-value **cannot go below ~0.25**
— significance was unreachable by construction, regardless of effect size.

Scoring per *utterance* (41 cases) fixed it, and it also matches the resampling
unit the statistics assume. The harness now resolves a real 10-point regression
at p=0.0005 while correctly calling a 0.6-point difference inconclusive.

### Gates that survive contact with a team

A gate that fires on run-to-run noise gets re-run until it goes green and then
ignored. So a regression that cannot be distinguished from noise is reported as a
**warning**, not a build failure. Absolute ceilings remain available for
requirements that are contractual rather than statistical.

---

## 6. Routing

Two stages, and the split is the design:

**Hard constraints eliminate.** Language support, streaming, word timestamps,
latency, cost, error-rate ceiling, breaker health. A 200 ms cap on live captions
is not a preference that a very cheap backend can outweigh by being cheap.

**Survivors are ranked** on quality/cost/latency, normalized *within the
candidate set* and weighted by the request's priority. Normalizing within the set
rather than against absolute scales makes the decision about the choice actually
available: when every candidate costs about the same, cost collapses to zero
influence and quality decides.

### Fallback

A retryable failure moves to the next candidate. A **non-retryable one stops the
chain immediately** — a malformed request will be malformed at the next provider
too, and retrying only spends the latency budget twice.

`total_deadline_ms` bounds the **whole chain**, not each attempt. Per-attempt
deadlines are how a three-provider fallback quietly turns a 1-second budget into
3 seconds of user-visible latency.

### Circuit breakers

On a realtime path, a 3-second timeout against a dead vendor *is* the entire
budget — the user waits 3 seconds to receive a transcript from the second
provider, on every utterance. The breaker exists to make that failure fast.

Only **retryable** failures open a breaker: a malformed-request error means this
request is broken, not the provider, and opening on it would pull a healthy
backend out of rotation. Half-open admits exactly one probe, because letting the
full backlog through the moment a service responds is how you knock it over
again.

### Unknown quality is treated as bad, not free

A provider with no measurement reports an error rate of 1.0. Otherwise a brand
new backend with no track record wins every quality-weighted decision by having
no recorded failures.

---

## 7. Realtime

### Endpointing is the latency floor

Wait too long and every utterance feels sluggish; cut too early and you truncate
someone mid-sentence and the ASR mis-recognizes the fragment. There is no free
option, so it is configuration rather than a buried constant.

The noise floor **adapts**, because a fixed threshold works in a quiet room and
fails in every real one — and it **stops adapting during speech**, or it climbs
to meet the speaker and the detector goes deaf partway through a sentence.

### Stabilization

A streaming ASR revises itself:

```
こんにちは → こんにちは今日 → こんにちは今日の議 → こんにちは本日の議題は
```

Rendering each hypothesis directly produces a line that rewrites itself several
times a second. That is not merely ugly — it is hard to read, because the eye
keeps re-reading a line that keeps changing. **A caption that flickers is worse
than one that lags.**

LocalAgreement commits only the prefix successive hypotheses agree on. Agreement
is computed on **tokens, not characters**: character-level comparison commits
half of a word the model is still deciding, and `食べ` / `食べない` differ only in
the part carrying the negation.

The invariant — *committed text is never revised* — required a real fix. The
first implementation compared prefix **lengths** only, so two hypotheses agreeing
on a longer but contradictory prefix could rewrite text the viewer had already
read. The agreed prefix must genuinely *extend* the committed one.

### What the session refuses to do

**It does not re-transcribe from the start.** Audio is released at each endpoint.
Retaining a whole meeting is ~115 MB per concurrent call at 16 kHz mono PCM16,
and re-decoding the session for every partial makes cost grow quadratically with
meeting length.

**It does not block audio intake on recognition.** `push_audio` buffers and
returns; ASR runs as a task. If recognition falls behind, audio still arrives and
the endpointer keeps working — dropping a caller's speech to wait on a slow model
is the one failure a voice product cannot recover from.

---

## 8. Citation-based verification

The failure mode is specific: an LLM will occasionally invent an action item
nobody agreed to, fluently enough that a reviewer skims past it. In a minutes
product, that invented task lands in someone's backlog with a due date.

The alternatives considered:

| Approach | Why not |
|---|---|
| LLM-as-judge | Doubles cost and latency, and introduces the judge's own uncertainty — now you have two models that can be confidently wrong |
| Embedding similarity | A fabricated claim about a meeting is *semantically similar* to the meeting. This is the case it is worst at |
| Human review | Correct, and does not scale to every meeting |

**Requiring a verbatim `source_quote` on every claim** turns the check into a
string operation: either the quote is in the transcript or it is not. No second
model, no additional latency, effectively zero cost.

It also improves the output on its own — a model that has to cite is less
inclined to embellish, because there is nowhere to put the embellishment.

Details that make it work in practice:

- Matching runs on **scoring-normalized** text, reusing the text layer. A model
  told to copy verbatim still differs in width and punctuation, and those are
  exactly the differences normalization exists to erase.
- Support is **scored**, not binary. Longest common substring, so a lightly
  reworded quote scores ~0.9 and a fabrication scores near 0. That distinction is
  what lets the system keep the first and drop the second.
- **Owners are validated against actual participants.** Assigning work to
  someone who was not in the meeting is a distinct failure from inventing the
  work, and the more awkward one.
- **Repair runs once**, naming the specific uncited quotes. "Try again" on a
  non-deterministic model is a coin flip; "these three quotes are not in the
  transcript" is a correctable instruction. A model that cannot cite on attempt
  two will not find a citation on attempt five, so a deterministic drop handles
  the rest without unbounded cost.
- **An empty transcript returns empty minutes without calling the model.**
  Asking a model to summarize silence is the classic way to get invention.

---

## 9. Operations

### Cost is measured where it is spent

A multi-model product has a cost structure that is hard to reconstruct from an
invoice: ASR bills per audio minute, the LLM per token, both with tiers, and the
mix shifts with whatever the router decided.

The ledger records every provider call as it happens, keyed by session, model and
tenant, and reports **cost per hour of audio** — the unit that divides into
revenue. Total spend says nothing without knowing how much audio produced it.

The offending call that breaks a budget is **still recorded**: money already
spent does not un-spend itself, and a ledger that drops it under-reports exactly
the incident worth investigating.

### Latency is a histogram

The mean latency of a voice pipeline describes nobody's experience — it is
dominated by the fast requests and hides the tail, so a p95 that doubled and a
mean that moved 4% look identical.

Export is CloudWatch EMF on stdout: no agent, no sidecar, no extra network path,
because on Fargate stdout already goes to CloudWatch Logs. Emitting a specific
JSON shape *is* the integration.

### Liveness and readiness are different questions

A saturated instance is alive but should stop receiving new sessions. `/health`
stays 200 (do not restart me) while `/ready` returns 503 (stop sending traffic).

A full server **rejects immediately** with close code 1013 rather than queueing.
A caller waiting behind a full server records audio nobody is transcribing and
would rather be told now.

### Three Terraform settings that are voice-specific

1. **ALB idle timeout is 15 minutes.** A WebSocket carrying a meeting is idle by
   the load balancer's definition whenever nobody is speaking; the 60 s default
   severs live calls.
2. **Deregistration delay exceeds graceful shutdown**, so a deploy drains
   in-flight sessions rather than severing them.
3. **Autoscaling tracks sessions per task, not CPU.** A streaming session spends
   most of its life awaiting a model, so CPU stays low while sockets and memory
   saturate — CPU-based scaling would under-provision exactly at capacity.

---

## 10. Testing strategy

Tests assert **behaviour with a stated reason**, not implementation. Most test
names read as a claim, and the docstring says why the claim matters:

```python
def test_a_word_far_from_every_turn_is_unknown_rather_than_guessed():
    """Confidently attributing a commitment to the wrong person is the worst outcome."""
```

Three deliberate choices:

**Mocks are infrastructure, not doubles.** Degradation is deterministic (hash of
salt+position, not a global RNG), so a backend with a *known* error rate exists to
validate the metrics that measure it. If CER cannot recover a known 8%
corruption, the bug is in the metric, not the model. There is a parametrized test
asserting exactly that round trip.

**The optional-dependency matrix is tested both ways.** Japanese tokenization is
optional, which creates a hazard specific to this project: a dev machine with
MeCab and a CI runner without it can silently disagree.
`tests/text/test_backend_parity.py` forces each backend explicitly and asserts
the scoring output is identical.

**CI smoke-tests the eval harness itself.** A clean backend must score 0 and a
degraded one must be caught as significant. The harness that guards quality needs
guarding too.

---

## 11. What I would do next

In rough priority order:

1. **Real audio.** The corpus is synthetic. The next step is a small set of real
   bilingual recordings with human reference transcripts, which would turn the
   error rates from "the harness works" into a benchmark.
2. **Streaming ASR for real.** The current streaming path re-decodes the
   in-progress utterance at an interval. A genuinely incremental decoder
   (faster-whisper with a sliding window, or a vendor streaming API) would cut
   both latency and cost substantially.
3. **Speaker enrolment.** Diarization currently produces anonymous labels. Mapping
   them to known participants — from a calendar invite, or voice enrolment — is
   what makes minutes attributable without a human relabelling step.
4. **Cross-instance sessions.** Sessions live in process today. A shared store
   would allow reconnection to a different task, removing the sticky-session
   requirement.
5. **Cost-aware partial scheduling.** Partial interval is fixed per session. It
   could adapt to speech rate and remaining budget — fewer interim decodes during
   a monologue, more during rapid exchange.
