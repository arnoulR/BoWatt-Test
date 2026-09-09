import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from providers import (
    FastEmbedSparseProvider,
    OpenAILLMProvider,
    ParallelWebProvider,
    canonicalize_url,
    deduplicate_queries,
    normalize_provider_error,
)


class ArrayLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_query_and_url_normalization() -> None:
    assert deduplicate_queries(["  Alpha   beta ", "alpha beta", "Gamma"]) == [
        "Alpha beta",
        "Gamma",
    ]
    assert canonicalize_url("HTTPS://Example.COM:443/a?z=2&a=1#part") == (
        "https://example.com/a?a=1&z=2"
    )
    assert canonicalize_url("http://user:secret@example.com") == ""
    assert canonicalize_url("file:///tmp/data") == ""


def test_provider_errors_are_concise_and_classified() -> None:
    error = RuntimeError("temporarily unavailable")
    error.status_code = 429

    normalized = normalize_provider_error("parallel", error)

    assert normalized.provider == "parallel"
    assert normalized.code == "429"
    assert normalized.retryable is True


@pytest.mark.asyncio
async def test_fastembed_uses_query_specific_embedding(tmp_path: Path) -> None:
    class FakeSparseModel:
        def query_embed(self, text: str) -> AsyncIterator[object]:
            assert text == "query"
            yield SimpleNamespace(
                indices=ArrayLike([1, 7]),
                values=ArrayLike([0.4, 0.8]),
            )

    provider = FastEmbedSparseProvider("fake", tmp_path)
    provider._model = FakeSparseModel()

    vector = await provider.embed_query("query")

    assert vector.indices == [1, 7]
    assert vector.values == [0.4, 0.8]


@pytest.mark.asyncio
async def test_parallel_reuses_session_and_keeps_partial_fetches() -> None:
    class FakeParallelClient:
        def __init__(self) -> None:
            self.search_kwargs = None
            self.extract_kwargs = None

        async def search(self, **kwargs):
            self.search_kwargs = kwargs
            return SimpleNamespace(
                session_id="session-1",
                results=[
                    SimpleNamespace(
                        url="https://EXAMPLE.com:443/a?b=2&a=1#x",
                        title="First",
                        publish_date="2026-01-01",
                        excerpts=["evidence"],
                    ),
                    SimpleNamespace(
                        url="https://example.com/a?a=1&b=2",
                        title="duplicate",
                        excerpts=["duplicate"],
                    ),
                ],
            )

        async def extract(self, **kwargs):
            self.extract_kwargs = kwargs
            return SimpleNamespace(
                session_id="session-1",
                results=[
                    SimpleNamespace(
                        url="https://example.com/a?a=1&b=2",
                        title="First",
                        excerpts=["full passage"],
                        publish_date=None,
                    )
                ],
                errors=[SimpleNamespace(error_type="blocked", content="second failed")],
            )

        async def close(self):
            pass

    client = FakeParallelClient()
    provider = ParallelWebProvider(
        api_key="test",
        model_name="fake-luna",
        max_retries=2,
        timeout_seconds=2,
        client=client,
    )

    results = await provider.search("objective", ["query", " Query "])
    fetched = await provider.fetch([results[0].canonical_url], "objective")

    assert len(results) == 1
    assert client.search_kwargs["search_queries"] == ["query"]
    assert client.search_kwargs["session_id"] is None
    assert client.extract_kwargs["session_id"] == "session-1"
    assert fetched.pages[0].passages == ["full passage"]
    assert fetched.errors == ["blocked: second failed"]


@pytest.mark.asyncio
async def test_shared_semaphore_bounds_parallel_outbound_calls() -> None:
    class CountingClient:
        active = 0
        maximum = 0

        async def search(self, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return SimpleNamespace(session_id="session", results=[])

    client = CountingClient()
    semaphore = asyncio.Semaphore(1)
    provider = ParallelWebProvider(
        api_key="test",
        model_name="fake-luna",
        max_retries=2,
        timeout_seconds=2,
        semaphore=semaphore,
        client=client,
    )

    await asyncio.gather(
        provider.search("objective", ["one"]),
        provider.search("objective", ["two"]),
    )

    assert client.maximum == 1


def test_openai_chat_configuration(monkeypatch) -> None:
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    OpenAILLMProvider(
        api_key="test",
        model_name="gpt-5.6-luna",
        reasoning_effort="low",
        max_retries=2,
        timeout_seconds=30,
    )

    assert captured == {
        "api_key": "test",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "use_responses_api": True,
        "store": False,
        "stream_usage": True,
        "max_retries": 2,
        "timeout": 30,
    }


@pytest.mark.asyncio
async def test_openai_provider_streams_prompt_and_usage() -> None:
    class FakeChat:
        def __init__(self) -> None:
            self.messages = []

        async def astream(self, messages):
            self.messages = messages
            yield SimpleNamespace(
                text="answer",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )

    chat = FakeChat()
    provider = OpenAILLMProvider(
        api_key="test",
        model_name="fake",
        reasoning_effort="low",
        max_retries=2,
        timeout_seconds=2,
        chat_model=chat,
    )

    chunks = [chunk async for chunk in provider.stream("system prompt", "user prompt")]

    assert chunks[0].text == "answer"
    assert chunks[0].usage.total_tokens == 7
    assert chat.messages[0].content == "system prompt"
    assert chat.messages[1].content == "user prompt"
