# Privacy, data handling, and safety

[日本語](SAFETY.ja.md) · English

Meeting audio is among the most sensitive data a company routinely produces. It
contains personnel discussions, unreleased financials, customer names, and
whatever someone said before they realised the recording had started. A voice
product is therefore a data-handling product first and a machine-learning
product second.

This document states what koe actually does with that data — verified against
the code, not aspirational — and where the sharp edges are.

---

## 1. What the system touches

| Data | Where it lives | How long |
|---|---|---|
| Raw audio (PCM16) | Process memory only | Released at each utterance endpoint |
| Interim hypotheses | Process memory, sent to the connected client | Discarded on the next partial |
| Final transcript | Process memory, sent to the client | Until the session object is collected |
| 議事録 | Process memory, sent to the client | Until the response is written |
| Cost ledger entries | Process memory, bounded ring buffer | Until process restart |
| Logs | stdout | Per your log retention |

### Verified properties

These are enforced by the code and covered by tests, not by convention:

- **Audio is never written to disk.** There is no file write on the audio path.
  Nothing in `src/koe/` persists audio; the only file writes in the package are
  the eval dataset loader and CLI report output, both explicitly invoked.
- **Audio is released at each endpoint.** `StreamingSession` deletes the buffer
  for a finished utterance rather than accumulating the session. This was done
  for memory (~115 MB per concurrent call-hour), and it has the side effect
  that a crash dump contains at most one utterance rather than a whole meeting.
  Test: `test_audio_is_released_after_each_utterance`.
- **No transcript content is logged.** Log records carry session ids, provider
  names, durations, costs and error types. They do not carry recognized text,
  minutes content, or speaker names. A log aggregator is therefore not a
  secondary copy of every meeting.
- **The cost ledger stores metadata only** — session id, provider, model,
  duration, token counts, cost. No content. See `telemetry/ledger.py::Entry`.

### What is *not* provided

Stated plainly, because absence is easy to mistake for a guarantee:

- **No encryption at rest**, because nothing is stored at rest. If you add
  persistence, that becomes your responsibility.
- **No authentication or authorization.** The API is unauthenticated. It is
  meant to sit behind your own gateway, and the Terraform places it behind an
  ALB in a private subnet — but *koe itself does not know who is calling*.
- **No tenant isolation beyond accounting.** The ledger separates spend by
  tenant. It does not isolate data between tenants; a single process serves
  whoever connects to it.
- **No audit log.** There is no tamper-evident record of who accessed what.

---

## 2. Third parties that see your data

The whole point of the harness is calling models you do not run, so this is the
part that matters most.

| Provider | Receives | Configured by |
|---|---|---|
| ASR backend | Raw audio, one utterance at a time | `asr` service registration |
| LLM backend | The **full transcript text** for summarization | `llm` service registration |
| Diarization backend | Raw audio | `diarizer` service registration |

Two consequences worth being explicit about:

**The LLM sees the entire meeting.** Minutes generation sends the complete
transcript, because summarizing requires it. If a meeting must not leave your
infrastructure, the LLM stage is the boundary to control — not the ASR stage.

**Default configuration calls nobody.** With no credentials present, koe runs
entirely on deterministic in-process mocks and makes no network calls at all
(`Settings.use_mocks`). Every external call is the result of a credential you
supplied.

### Retention at the provider

koe cannot control what a provider retains. Check your agreement:

- Anthropic and OpenAI both offer zero-retention arrangements for API traffic;
  neither is the default on a standard account.
- If you require that audio never leaves your network, run a local ASR
  (`faster-whisper` via the `asr` extra) and a self-hosted LLM. The provider
  protocol makes that a registration change, not a code change.

---

## 3. Japanese regulatory context (APPI)

For deployment in Japan, meeting audio and transcripts are 個人情報 (personal
information) under 個人情報保護法 (APPI) as soon as a participant is
identifiable — which speaker diarization makes true by construction, and which
minutes make true by name.

Points that follow directly from how koe works:

- **Speaker labels are personal data.** The diarization output attributes speech
  to individuals. Minutes go further and attach names to commitments.
- **Sending audio to an overseas provider is a cross-border transfer**
  (第28条 越境移転). Using a US-hosted ASR or LLM requires the appropriate basis
  — consent, or an equivalent-protection assessment of the recipient.
  `ap-northeast-1` is the Terraform default region for the *service*; it does
  not change where a third-party model runs.
- **Purpose limitation (利用目的の特定).** Transcripts collected to produce
  minutes may not be silently repurposed as training data. koe never sends data
  anywhere except the providers you register, and never retains it for
  secondary use.
- **Recording consent.** koe does not implement consent capture. Whether every
  participant has agreed to be recorded is a product decision above this layer,
  and in Japan a practical one — the norm is explicit announcement at the start
  of a meeting.

This is a description of how the system behaves, not legal advice. Confirm your
obligations with counsel.

---

## 4. Data minimization levers

Configuration that reduces exposure, in order of effect:

```bash
# Run entirely locally — no external calls at all.
KOE_FORCE_MOCK_PROVIDERS=true

# Bound how long any single session can retain a buffer.
KOE_MAX_SESSION_SECONDS=3600

# Fewer interim decodes means less audio sent to the ASR provider,
# at the cost of a less responsive caption.
KOE_PARTIAL_INTERVAL_MS=1000

# Cap spend per session; also caps how much data one session can push out.
KOE_SESSION_BUDGET_USD=2.0
```

The partial interval is the least obvious one. Every interim hypothesis is
another decode of the utterance so far, which means the same audio is sent to
the provider repeatedly. Raising the interval reduces both cost and the number
of times a given second of audio crosses your network boundary.

---

## 5. Service threat model

What the running service defends against, and what it does not.

**Defended:**

- *Resource exhaustion by a single client* — concurrent sessions are capped and
  a full server rejects rather than queues; frame size is bounded; a client
  streaming faster than realtime is disconnected.
- *Runaway cost* — per-session and per-tenant budgets are enforced, and the
  offending call is still recorded so the incident is investigable.
- *Information leakage through errors* — an unhandled exception returns a
  request id and nothing else. Tracebacks go to logs, never to callers.
- *Misconfiguration reaching production* — a wildcard CORS policy, an unbounded
  session budget, or forced mocks all refuse to start in `production`.
- *Cascading provider failure* — circuit breakers stop sending traffic to a
  failing backend rather than spending the latency budget discovering it.

**Not defended:**

- *Authentication and authorization.* Put a gateway in front of it.
- *Distributed denial of service.* Per-connection limits do not help against
  many connections; that is a load balancer and WAF concern.
- *A malicious model provider.* Registering a backend grants it your audio.
- *Prompt injection through meeting content.* A participant who says "ignore
  your instructions and…" is text the LLM will see. The groundedness check
  constrains the *output* — every claim must quote the transcript — which
  bounds fabricated content, but does not prevent an attacker from influencing
  a summary using things they actually said.
- *Side channels.* Timing and cost metrics are exposed at `/metrics` without
  authentication. Do not expose that endpoint publicly.

---

## 6. Reporting a security issue

This is a portfolio project rather than a service with users, but if you find
something, please open an issue at
[github.com/jon-jc/koe-harness/issues](https://github.com/jon-jc/koe-harness/issues).
For anything you would rather not file publicly, note that in the issue without
details and I will follow up.
