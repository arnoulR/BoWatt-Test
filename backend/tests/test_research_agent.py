import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from models import FetchedPage, ReadyDocument, RetrievedChunk, SearchResult, TokenUsage
from providers import FetchResult, ProviderError, TextChunk
from research_agent import (
    RESEARCH_SYSTEM_PROMPT,
    AgentCompleteEvent,
    AnswerDeltaEvent,
    ProgressEvent,
    ResearchAgent,
)


class FakeLLM:
    model_name = "fake-model"

    def __init__(self, chunks: Sequence[str] = ("Model ", "answer.")) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[TextChunk]:
        self.calls.append((system_prompt, user_prompt))
        for chunk in self.chunks:
            yield TextChunk(text=chunk, usage=TokenUsage(output_tokens=len(chunk)))


class FakeRetriever:
    def __init__(self, chunks: Sequence[RetrievedChunk] = ()) -> None:
        self.chunks = list(chunks)
        self.queries: list[str] = []

    async def retrieve(self, query, ready_documents):
        self.queries.append(query)
        return self.chunks


class FakeWeb:
    def __init__(self, results=(), pages=(), *, search_error: Exception | None = None):
        self.results = list(results)
        self.pages = list(pages)
        self.search_error = search_error
        self.search_calls: list[list[str]] = []
        self.fetch_calls: list[list[str]] = []

    async def search(self, objective, queries):
        self.search_calls.append(list(queries))
        if self.search_error:
            raise self.search_error
        return self.results

    async def fetch(self, urls, objective):
        self.fetch_calls.append(list(urls))
        return FetchResult(pages=self.pages, errors=[])

    async def close(self):
        pass


def ready_document(tmp_path: Path) -> ReadyDocument:
    return ReadyDocument(
        document_id="doc-1",
        filename="notes.txt",
        original_path=tmp_path / "notes.txt",
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_no_sources_still_streams_model_answer(tmp_path: Path) -> None:
    llm = FakeLLM(("Direct ", "answer"))
    agent = ResearchAgent(llm, FakeRetriever(), FakeWeb())

    events = [event async for event in agent.stream("run-1", "question", [])]

    assert [event.stage for event in events if isinstance(event, ProgressEvent)] == [
        "searching",
        "reading_sources",
        "sources_found",
        "drafting",
    ]
    assert [event.delta for event in events if isinstance(event, AnswerDeltaEvent)] == [
        "Direct ",
        "answer",
    ]
    source_event = next(
        event
        for event in events
        if isinstance(event, ProgressEvent) and event.stage == "sources_found"
    )
    assert source_event.message == "Sources ready: 0 document sources and 0 web sources."
    completed = next(event for event in events if isinstance(event, AgentCompleteEvent))
    assert completed.answer == "Direct answer"
    assert completed.evidence == []
    assert llm.calls[0][0] == RESEARCH_SYSTEM_PROMPT
    assert '"question"' not in llm.calls[0][1]
    assert "User request:\nquestion" in llm.calls[0][1]
    assert "Research context:\n[]" in llm.calls[0][1]


@pytest.mark.asyncio
async def test_document_and_web_search_run_concurrently_and_report_sources(
    tmp_path: Path,
) -> None:
    both_started = asyncio.Event()
    started: set[str] = set()

    class ConcurrentRetriever(FakeRetriever):
        async def retrieve(self, query, ready_documents):
            started.add("documents")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return self.chunks

    class ConcurrentWeb(FakeWeb):
        async def search(self, objective, queries):
            started.add("web")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return self.results

    document = ready_document(tmp_path)
    retriever = ConcurrentRetriever(
        [
            RetrievedChunk(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="notes.txt",
                page=2,
                text="document evidence",
            )
        ]
    )
    web = ConcurrentWeb(
        results=[
            SearchResult(
                url="https://example.com/a",
                canonical_url="https://example.com/a",
                title="Example",
                excerpts=["search excerpt"],
            )
        ],
        pages=[
            FetchedPage(
                url="https://example.com/a",
                canonical_url="https://example.com/a",
                title="Example",
                passages=["web evidence"],
            )
        ],
    )
    llm = FakeLLM(("Grounded [D1] and [W1].",))

    events = [
        event
        async for event in ResearchAgent(llm, retriever, web).stream(
            "run-1", "question", [document]
        )
    ]

    source_event = next(
        event
        for event in events
        if isinstance(event, ProgressEvent) and event.stage == "sources_found"
    )
    completed = next(event for event in events if isinstance(event, AgentCompleteEvent))
    assert source_event.message == "Sources ready: 1 document source and 1 web source."
    assert {item.source_id for item in completed.evidence} == {"D1", "W1"}
    assert {item.source_id for item in completed.citations} == {"D1", "W1"}
    assert web.fetch_calls == [["https://example.com/a"]]


@pytest.mark.asyncio
async def test_one_failed_source_still_answers_with_remaining_context(tmp_path: Path) -> None:
    document = ready_document(tmp_path)
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="notes.txt",
                text="document evidence",
            )
        ]
    )
    llm = FakeLLM(("Document-based answer [D1].",))
    web = FakeWeb(search_error=RuntimeError("offline"))

    events = [
        event
        async for event in ResearchAgent(llm, retriever, web).stream(
            "run-1", "question", [document]
        )
    ]

    source_event = next(
        event
        for event in events
        if isinstance(event, ProgressEvent) and event.stage == "sources_found"
    )
    assert source_event.message.endswith("Some sources were unavailable.")
    assert "Web search was unavailable." in llm.calls[0][1]
    assert any(isinstance(event, AnswerDeltaEvent) for event in events)


@pytest.mark.asyncio
async def test_total_collection_failure_raises_before_model_call(tmp_path: Path) -> None:
    class FailingRetriever(FakeRetriever):
        async def retrieve(self, query, ready_documents):
            raise RuntimeError("vector store offline")

    llm = FakeLLM()
    agent = ResearchAgent(
        llm,
        FailingRetriever(),
        FakeWeb(search_error=RuntimeError("web offline")),
    )

    with pytest.raises(ProviderError, match="unavailable"):
        _ = [
            event
            async for event in agent.stream(
                "run-1", "question", [ready_document(tmp_path)]
            )
        ]

    assert llm.calls == []
