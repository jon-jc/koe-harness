# Third-party notices

koe is MIT licensed. It depends on the following third-party software, each
under its own license. This list covers direct dependencies; run
`pip-licenses` or `npm ls` for the full transitive tree of a given install.

## Python — always installed

| Package | License | Purpose |
|---|---|---|
| [pydantic](https://github.com/pydantic/pydantic) | MIT | Domain model validation and serialization |
| [pydantic-settings](https://github.com/pydantic/pydantic-settings) | MIT | Environment-driven configuration |
| [typing-extensions](https://github.com/python/typing_extensions) | PSF-2.0 | Backported typing constructs |

## Python — optional extras

| Package | Extra | License | Purpose |
|---|---|---|---|
| [fastapi](https://github.com/fastapi/fastapi) | `api` | MIT | HTTP and WebSocket surface |
| [uvicorn](https://github.com/encode/uvicorn) | `api` | BSD-3-Clause | ASGI server |
| [websockets](https://github.com/python-websockets/websockets) | `api` | BSD-3-Clause | WebSocket protocol |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) | `llm` | MIT | Claude client |
| [openai](https://github.com/openai/openai-python) | `llm` | Apache-2.0 | GPT client |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `asr` | MIT | Local speech recognition |
| [soundfile](https://github.com/bastibe/python-soundfile) | `asr` | BSD-3-Clause | Audio file I/O |
| [numpy](https://github.com/numpy/numpy) | `asr` | BSD-3-Clause | Array operations |
| [fugashi](https://github.com/polm/fugashi) | `ja` | MIT | MeCab bindings for Japanese tokenization |
| [unidic-lite](https://github.com/polm/unidic-lite) | `ja` | BSD-3-Clause | Japanese dictionary for MeCab |
| [boto3](https://github.com/boto/boto3) | `aws` | Apache-2.0 | AWS SDK |
| [aws-lambda-powertools](https://github.com/aws-powertools/powertools-lambda-python) | `aws` | MIT | AWS observability helpers |
| [typer](https://github.com/fastapi/typer) | `cli` | MIT | Command line interface |
| [rich](https://github.com/Textualize/rich) | `cli` | MIT | Terminal rendering |

> **Note on MeCab.** `fugashi` links against [MeCab](https://taku910.github.io/mecab/),
> which is triple-licensed under GPL, LGPL, and BSD. `unidic-lite` bundles a
> UniDic dictionary under BSD-3-Clause. koe uses fugashi through its Python
> API only, and treats it as an optional dependency — the package functions
> without it, falling back to character segmentation.

## Python — development only

| Package | License |
|---|---|
| [pytest](https://github.com/pytest-dev/pytest) | MIT |
| [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) | Apache-2.0 |
| [pytest-cov](https://github.com/pytest-dev/pytest-cov) | MIT |
| [ruff](https://github.com/astral-sh/ruff) | MIT |
| [mypy](https://github.com/python/mypy) | MIT |
| [hypothesis](https://github.com/HypothesisWorks/hypothesis) | MPL-2.0 |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause |

## TypeScript — development only

The shipped browser bundle has **no runtime dependencies**; these are build
tools and their output is the bundle itself.

| Package | License | Purpose |
|---|---|---|
| [esbuild](https://github.com/evanw/esbuild) | MIT | Bundler |
| [typescript](https://github.com/microsoft/TypeScript) | Apache-2.0 | Type checker |

## Adapted source

Not a dependency, and not merely an influence: code here was **derived from**
the following, which is why its license text is reproduced in full.

### OpenWhispr

<https://github.com/OpenWhispr/openwhispr> — MIT License, Copyright (c) 2024
OpenWhispr Team.

Three pieces of OpenWhispr's dictation workflow were adapted for koe. In each
case what carried over is a rule about text and the reasoning behind it,
reimplemented in Python against koe's own script analysis:

| koe | from | what was taken |
|---|---|---|
| `src/koe/text/spacing.py` | `src/helpers/smartSpacing.js` | The codepoint ranges for scripts that do not separate words, and the decision to exclude Hangul from them |
| `src/koe/text/thinking.py` | `src/helpers/stripThinking.js` | The three-pass ordering: innermost closed pairs, then an unterminated trailing block, then orphan closing tags |
| `src/koe/text/vocabulary.py` | the user dictionary and `src/helpers/dictionaryImport.js` | A user-editable word list as the answer to proper nouns, and the lenient import format |

Where koe departs, the module docstring says so and why. The two substantial
departures: spacing is decided from both sides of a seam rather than the left
side alone, because koe holds both fragments where a paste-at-cursor does not;
and vocabulary matching changes strategy by script, because word-boundary
anchoring cannot work in Japanese.

```
MIT License

Copyright (c) 2024 OpenWhispr Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### DeepSeek Harness

<https://github.com/deepseek-ai/deepseek-harness> — MIT License, Copyright (c)
2026 DeepSeek.

`src/koe/harness/` is a port of the parts of dsh that make a chat surface a
harness. dsh is a TypeScript monorepo built on Cordis and koe's backend is
Python, so no file is copied: the architecture, the event taxonomy and the
scheduling algorithm are dsh's, reimplemented against koe's kernel, tool
registry and providers.

| koe | from | what was taken |
|---|---|---|
| `harness/session.py` | `packages/core/session` | The append-only session log, the `turn/*` `step/*` `tool/*` event taxonomy, and deriving message history by folding the log rather than storing it |
| `harness/inbox.py` | `agent-loop/src/inbox.ts` | Two pending queues, mutation as durable splices, and atomic claiming at turn and step boundaries — the mechanism behind steering |
| `harness/scheduler.py` | `agent-loop/src/tool-calls.ts` | Exclusive barriers, the bounded parallel pool, reclassification as the pool fills, model-ordered commit of out-of-order completions, and synthetic `aborted before dispatch` results |
| `harness/agent.py` | `agent-loop/src/agent.ts` | The turn/step machine, the queue-consuming driver, explicit cooperative cancellation, and committing streamed text as interrupted |
| `harness/compaction.py` | `compaction/`, `compaction-basic/`, `compaction-tool-result-pruner/` | Shadowing rather than deleting, replacements landing at the position they replace, tool-pairing balance as the legality rule for a cut, the retain-tail region selection, the start/summary/replacement/end bracket, and the model-free tool-result pruner |
| `harness/prompt.py` | `core/system-prompt` | Named, centrally ordered prompt sections contributed as disposable effects; deterministic name tie-break; `complete` sections; refusing assembly on an unresolved variable |
| `harness/commands.py` | the command seam and `compaction/command-compact` | Commands running instead of turns, and outcomes that say what happened *to the conversation* rather than that something failed |
| `harness/tokens.py` | `token-meter` (seam only) | The idea of a replaceable metering seam. The bilingual estimator is koe's own — dsh's implementation is not in the public tree |

The event names are dsh's spelling on purpose. Inventing koe's own would have
made the two implementations impossible to compare, which is the main thing a
reader of both would want to do.

**Not ported**, and absent rather than stubbed: the projection registry,
persistence backends, delegation and sub-agents, ACP, code-runtime and e2b
sandboxes, attachments, and dsh's request-freeze provenance. koe's chat is one
agent in one process.

```
MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Prior art and design influence

Not dependencies and not adapted source — but the design owes them a citation:

- **[Cordis](https://github.com/cordiverse/cordis)** (MIT) — the plugin
  kernel's composability model: scoped lifetimes, dependency-gated activation,
  and spatial filters. Reimplemented in Python for a realtime audio domain,
  by way of dsh above.
- **LocalAgreement** — the partial-stabilization policy, as described by
  Macháček, Dabre and Bojar in *Turning Whisper into Real-Time Transcription
  System* (2023) and used in `whisper_streaming`.
