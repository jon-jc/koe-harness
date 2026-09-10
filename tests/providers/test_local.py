"""Local inference: endpoints, discovery, the LLM client, and Whisper's priors.

No network. Every test that would reach a server reaches a fake one instead,
because the thing worth testing is koe's handling of what local servers
actually return -- and the interesting returns are the malformed ones, which a
healthy Ollama will never produce on demand.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from koe.agent.adapters import LocalAdapter, _local_reply, _rejected_tools, adapter_for
from koe.agent.loop import ChatMessage
from koe.providers.base import ProviderError
from koe.providers.local import discovery
from koe.providers.local.endpoints import (
    SERVERS_BY_ID,
    is_loopback,
    normalize_base_url,
    openai_base_candidates,
)
from koe.providers.local.http import LocalHTTPError
from koe.providers.local.llm import LocalLLM, _first_message_text
from koe.providers.local.whisper import DEFAULT_SIZE, SIZES, SIZES_BY_ID, usable_for
from koe.text.script import Language

# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:1234/v1/", "http://127.0.0.1:1234/v1"),
        ("  http://localhost:8080  ", "http://localhost:8080"),
        ("http://host:1234/v1?key=x#frag", "http://host:1234/v1"),
        ("", ""),
        ("http://", ""),
    ],
)
def test_a_base_url_is_cleaned_up(typed: str, expected: str) -> None:
    """A scheme is added because every one of these projects documents its
    address without one, and `localhost:11434` is not a valid URL."""
    assert normalize_base_url(typed) == expected


def test_a_bare_origin_also_gets_tried_with_v1() -> None:
    """People paste what the docs show, which is the origin."""
    assert openai_base_candidates("localhost:11434") == (
        "http://localhost:11434",
        "http://localhost:11434/v1",
    )


def test_a_base_that_already_ends_in_v1_is_left_alone() -> None:
    assert openai_base_candidates("http://localhost:1234/v1") == ("http://localhost:1234/v1",)


def test_lm_studios_native_rest_base_maps_to_its_openai_sibling() -> None:
    """/api/v0 is what LM Studio's own UI shows; /v1 is where chat lives."""
    assert openai_base_candidates("http://localhost:1234/api/v0") == (
        "http://localhost:1234/api/v0",
        "http://localhost:1234/v1",
    )


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:1234", True),
        ("http://[::1]:8080", True),
        ("http://192.168.1.50:11434", False),
        ("https://api.example.com", False),
    ],
)
def test_loopback_is_distinguished_from_the_rest_of_the_network(url: str, local: bool) -> None:
    """koe may only claim "never leaves this device" for the first kind. A model
    server on the LAN is a fine deployment and a different promise."""
    assert is_loopback(url) is local


def test_ollama_is_probed_on_its_documented_port() -> None:
    assert SERVERS_BY_ID["ollama"].port == 11434
    assert SERVERS_BY_ID["ollama"].base_url == "http://127.0.0.1:11434/v1"


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_ollamas_native_listing_carries_the_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/tags reports parameters and quantization; /v1/models does not, and
    that is the number that says whether a model fits in RAM."""
    payload = {
        "models": [
            {
                "name": "qwen2.5:7b",
                "size": 4_700_000_000,
                "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
            }
        ]
    }
    models = discovery._models_from_ollama(payload)
    assert models[0].id == "qwen2.5:7b"
    assert models[0].parameters == "7.6B"
    assert models[0].quantization == "Q4_K_M"


def test_a_malformed_listing_yields_nothing_rather_than_raising() -> None:
    """A local server is whatever the user installed, at whatever version."""
    assert discovery._models_from_ollama({"models": "not a list"}) == ()
    assert discovery._models_from_ollama({}) == ()
    assert discovery._models_from_openai({"data": [{"no_id": True}]}) == ()
    assert discovery._models_from_openai("nonsense") == ()


async def test_a_running_server_with_no_models_is_reported_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LM Studio running with nothing loaded has a different fix from LM Studio
    not running, and collapsing them sends someone to reinstall working
    software."""

    async def answer_but_empty(url: str, **_: Any) -> Any:
        return {"data": []}

    monkeypatch.setattr(discovery, "get_json", answer_but_empty)

    found = await discovery.probe(SERVERS_BY_ID["lmstudio"])

    assert found is not None
    assert found.ready is False
    assert "no models" in found.note


async def test_a_server_that_is_not_running_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(url: str, **_: Any) -> Any:
        raise LocalHTTPError("connection refused")

    monkeypatch.setattr(discovery, "get_json", refused)

    assert await discovery.probe(SERVERS_BY_ID["lmstudio"]) is None


async def test_one_misbehaving_server_does_not_cost_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe is best-effort by definition."""

    async def explode_on_ollama(url: str, **_: Any) -> Any:
        if "11434" in url:
            raise RuntimeError("something unexpected")
        if "1234" in url:
            return {"data": [{"id": "local-model"}]}
        raise LocalHTTPError("connection refused")

    monkeypatch.setattr(discovery, "get_json", explode_on_ollama)

    sweep = await discovery.discover()

    assert sweep.any_models
    assert [s.server_id for s in sweep.servers] == ["lmstudio"]
    assert "ollama" in {s.id for s in sweep.absent}


# --------------------------------------------------------------------------
# the LLM client
# --------------------------------------------------------------------------


def test_the_chat_endpoint_settles_the_v1_question() -> None:
    assert (
        LocalLLM(model="m", base_url="localhost:11434").endpoint
        == "http://localhost:11434/v1/chat/completions"
    )
    assert (
        LocalLLM(model="m", base_url="http://localhost:1234/v1").endpoint
        == "http://localhost:1234/v1/chat/completions"
    )


def test_a_local_model_costs_nothing_and_says_so() -> None:
    info = LocalLLM(model="qwen2.5:7b", base_url="localhost:11434").info
    assert info.cost_per_1k_input_tokens_usd == 0.0
    assert info.cost_per_1k_output_tokens_usd == 0.0


def test_a_local_model_declares_its_disadvantages() -> None:
    """Zero cost with no counterweight would win every routing decision, and
    koe would feel broken rather than free."""
    info = LocalLLM(model="qwen2.5:7b", base_url="localhost:11434").info
    assert info.typical_first_result_ms > 1_000.0
    # Japanese is where an open-weights model's mostly-English training mix
    # shows most, so the prior must be worse there than for English.
    assert info.error_rate_for(Language.JA) > info.error_rate_for(Language.EN)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"choices": [{"message": {"content": "hello"}}]}, "hello"),
        ({"choices": [{"text": "older shape"}]}, "older shape"),
        (
            {"choices": [{"message": {"content": [{"type": "text", "text": "parts"}]}}]},
            "parts",
        ),
        ({"choices": []}, ""),
        ({}, ""),
        ("not json at all", ""),
    ],
)
def test_the_reply_shapes_local_servers_actually_emit(body: Any, expected: str) -> None:
    assert _first_message_text(body) == expected


async def test_an_unreachable_server_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalLLM(model="m", base_url="localhost:11434")

    async def refused(*_: Any, **__: Any) -> Any:
        raise LocalHTTPError("connection refused")

    monkeypatch.setattr("koe.providers.local.llm.post_json", refused)

    with pytest.raises(ProviderError, match="no local model server answered"):
        await provider.complete([])


async def test_a_404_names_the_actual_mistake(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base URL being the web UI's origin rather than the API's is the most
    common local misconfiguration by a distance."""
    provider = LocalLLM(model="m", base_url="http://localhost:1234/v1")

    async def not_found(*_: Any, **__: Any) -> Any:
        raise LocalHTTPError("404 from ...", status=404)

    monkeypatch.setattr("koe.providers.local.llm.post_json", not_found)

    with pytest.raises(ProviderError, match="/v1"):
        await provider.complete([])


async def test_reasoning_blocks_are_stripped_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local models are disproportionately reasoning distills; it is what fits
    on a laptop."""
    provider = LocalLLM(model="m", base_url="localhost:11434")

    async def thinks(*_: Any, **__: Any) -> Any:
        return {"choices": [{"message": {"content": "<think>hmm</think>Q3 は好調です。"}}]}

    monkeypatch.setattr("koe.providers.local.llm.post_json", thinks)

    assert (await provider.complete([])).text == "Q3 は好調です。"


# --------------------------------------------------------------------------
# the agent adapter
# --------------------------------------------------------------------------


def test_a_local_provider_gets_the_local_adapter() -> None:
    """Dispatch cannot be on the name alone: a local provider is labelled with
    the server it points at, so "ollama" would fall through to the mock."""
    provider = LocalLLM(model="m", base_url="localhost:11434", label="ollama")
    assert isinstance(adapter_for(provider), LocalAdapter)


def test_tool_arguments_are_accepted_parsed_or_as_a_string() -> None:
    """OpenAI sends a JSON string. Several local servers send the object,
    having parsed it to validate against the schema."""
    as_string = _local_reply(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "t", "arguments": '{"a": 1}'}}
                        ]
                    }
                }
            ]
        }
    )
    as_object = _local_reply(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "t", "arguments": {"a": 1}}}
                        ]
                    }
                }
            ]
        }
    )
    assert as_string.tool_calls[0].arguments == {"a": 1}
    assert as_object.tool_calls[0].arguments == {"a": 1}


def test_invalid_tool_arguments_do_not_crash_the_turn() -> None:
    """An empty dict lets the tool reject it with a message the model can act
    on, where a decode error ends the turn."""
    reply = _local_reply(
        {
            "choices": [
                {"message": {"tool_calls": [{"function": {"name": "t", "arguments": "{oops"}}]}}
            ]
        }
    )
    assert reply.tool_calls[0].arguments == {}
    # The id was missing too; it is synthesized so the result can quote it back.
    assert reply.tool_calls[0].id


async def test_a_model_that_cannot_call_tools_still_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local tool support ranges from complete to absent, and the same server
    answers differently depending on which model is loaded. Failing the turn
    when the question needed no tools is the wrong outcome."""
    provider = LocalLLM(model="m", base_url="localhost:11434")
    adapter = LocalAdapter(provider)
    attempts: list[bool] = []

    async def rejects_tools(url: str, payload: dict[str, Any], **_: Any) -> Any:
        sent = "tools" in payload
        attempts.append(sent)
        if sent:
            raise LocalHTTPError("400: this model does not support tools", status=400)
        return {"choices": [{"message": {"content": "answered anyway"}}]}

    monkeypatch.setattr("koe.agent.adapters.post_json", rejects_tools)

    tools = [{"name": "t", "description": "d", "parameters": {}}]
    reply = await adapter.reply([ChatMessage(role="user", content="hi")], system="s", tools=tools)

    assert reply.text == "answered anyway"
    assert attempts == [True, False]
    assert adapter.tools_supported is False


async def test_the_lack_of_tool_support_is_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-learning it every turn would double the latency of the slow path."""
    provider = LocalLLM(model="m", base_url="localhost:11434")
    adapter = LocalAdapter(provider)
    attempts: list[bool] = []

    async def rejects_tools(url: str, payload: dict[str, Any], **_: Any) -> Any:
        sent = "tools" in payload
        attempts.append(sent)
        if sent:
            raise LocalHTTPError("400: unsupported tool call", status=400)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("koe.agent.adapters.post_json", rejects_tools)

    tools = [{"name": "t", "description": "d", "parameters": {}}]
    messages = [ChatMessage(role="user", content="hi")]
    await adapter.reply(messages, system="s", tools=tools)
    attempts.clear()
    await adapter.reply(messages, system="s", tools=tools)

    assert attempts == [False]


async def test_a_failure_unrelated_to_tools_is_not_silently_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying an out-of-memory error without tools would report a confusing
    success or a second, identical failure."""
    provider = LocalLLM(model="m", base_url="localhost:11434")
    adapter = LocalAdapter(provider)
    calls = 0

    async def out_of_memory(url: str, payload: dict[str, Any], **_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise LocalHTTPError("500: not enough memory to load model", status=500)

    monkeypatch.setattr("koe.agent.adapters.post_json", out_of_memory)

    with pytest.raises(ProviderError):
        await adapter.reply(
            [ChatMessage(role="user", content="hi")],
            system="s",
            tools=[{"name": "t", "description": "d", "parameters": {}}],
        )
    assert calls == 1


def test_only_a_server_that_answered_can_be_refusing_tools() -> None:
    """An unreachable server has not told us anything about tool support."""
    assert _rejected_tools(LocalHTTPError("refused")) is False
    assert _rejected_tools(LocalHTTPError("400: unknown field 'tools'", status=400)) is True


# --------------------------------------------------------------------------
# Whisper's size priors
# --------------------------------------------------------------------------


def test_the_small_checkpoints_are_not_offered_for_japanese() -> None:
    """Whisper's training mix is overwhelmingly English, and the small
    checkpoints spend what multilingual capacity they have on languages closer
    to it. Presenting these as a neutral speed slider would mislead in exactly
    the case koe exists for."""
    assert not usable_for("tiny", Language.JA)
    assert not usable_for("base", Language.JA)
    assert not usable_for("small", Language.JA)
    assert usable_for("medium", Language.JA)
    assert usable_for("large-v3", Language.JA)


def test_english_tolerates_a_smaller_checkpoint_than_japanese_does() -> None:
    assert usable_for("small", Language.EN)
    assert not usable_for("small", Language.JA)


def test_a_mixed_meeting_needs_a_checkpoint_fluent_in_both() -> None:
    assert not usable_for("small", Language.MIXED)
    assert usable_for("large-v3", Language.MIXED)


def test_japanese_is_worse_at_every_size() -> None:
    for size in SIZES:
        assert size.error_rate[Language.JA] > size.error_rate[Language.EN], size.id


def test_bigger_checkpoints_are_better_and_slower() -> None:
    """The trade the size list exists to express. Turbo is excluded: being off
    that curve is the entire reason it is the default."""
    curve = [s for s in SIZES if s.id != "large-v3-turbo"]
    for earlier, later in itertools.pairwise(curve):
        assert later.error_rate[Language.JA] < earlier.error_rate[Language.JA]
        assert later.typical_rtf > earlier.typical_rtf


def test_the_default_is_suitable_for_japanese() -> None:
    """koe is a bilingual tool; a default that cannot do Japanese is not one."""
    assert usable_for(DEFAULT_SIZE, Language.JA)
    assert usable_for(DEFAULT_SIZE, Language.MIXED)


def test_the_default_is_cheaper_than_the_best_checkpoint() -> None:
    """Turbo rather than large-v3 is the difference between usable on a laptop
    and not."""
    assert SIZES_BY_ID[DEFAULT_SIZE].typical_rtf < SIZES_BY_ID["large-v3"].typical_rtf


def test_an_unknown_size_is_not_recommended_for_anything() -> None:
    assert not usable_for("gpt-4", Language.EN)
