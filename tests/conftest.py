"""Suite-wide setup.

Set before anything imports koe, because `Settings` reads the environment at
construction and several modules build one at import time.
"""

from __future__ import annotations

import os

# Discovery probes the well-known local-model ports at startup. Left on, the
# suite's behaviour would depend on whether the machine running it happens to
# have Ollama open -- tests that assert the mock LLM answers would get a real
# local model on a developer's laptop and the mock on CI. An explicit
# `local_llm_base_url` is still honoured, which is how the local tests
# exercise the path deliberately.
os.environ.setdefault("KOE_DISCOVER_LOCAL_MODELS", "false")
