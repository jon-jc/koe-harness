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

## Prior art and design influence

Not dependencies — no code is copied from either — but the design owes them a
citation:

- **[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)** (MIT)
  and **[Cordis](https://github.com/cordiverse/cordis)** (MIT) — the plugin
  kernel's composability model: scoped lifetimes, dependency-gated activation,
  and spatial filters. Reimplemented in Python for a realtime audio domain.
- **LocalAgreement** — the partial-stabilization policy, as described by
  Macháček, Dabre and Bojar in *Turning Whisper into Real-Time Transcription
  System* (2023) and used in `whisper_streaming`.
