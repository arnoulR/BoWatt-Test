import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from database import ResearchDatabase
from models import (
    ResearchAnswerDeltaEvent,
    ResearchCompleteEvent,
    ResearchErrorEvent,
    ResearchProgressEvent,
    ResearchRunStatus,
    TokenUsage,
)
from providers import FetchResult, TextChunk
from research_service import ResearchService


class FakeDocumentDatabase:
    async def list_ready(self):
        return []


class UnusedRetriever:
    async def retrieve(self, query, ready_documents):
        raise AssertionError("retrieval should not run without documents")


class FakeWeb:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    async def search(self, objective, queries):
        if self.error:
            raise self.error
        return []

    async def fetch(self, urls, objective):
        return FetchResult(pages=[], errors=[])

    async def close(self):
        self.closed = True


class FakeLLM:
    model_name = "fake-model"

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[TextChunk]:
        yield TextChunk(text="model ", usage=TokenUsage(output_tokens=1))
        yield TextChunk(text="answer", usage=TokenUsage(output_tokens=2))


def build_service(path: Path, llm, web: FakeWeb) -> ResearchService:
    return ResearchService(
        document_database=FakeDocumentDatabase(),
        research_database=ResearchDatabase(path),
        llm=llm,
        retriever=UnusedRetriever(),
        web_factory=lambda: web,
    )


@pytest.mark.asyncio
async def test_service_streams_and_persists_model_answer(tmp_path: Path) -> None:
    web = FakeWeb()
    service = build_service(tmp_path / "research.sqlite3", FakeLLM(), web)
    await service.start()
    run_id = await service.create_run("question")

    events = [event async for event in service.stream_run(run_id, "question")]
    run = await service.get_run(run_id)

    assert [event.stage for event in events if isinstance(event, ResearchProgressEvent)] == [
        "started",
        "searching",
        "reading_sources",
        "sources_found",
        "drafting",
    ]
    assert "".join(
        event.delta for event in events if isinstance(event, ResearchAnswerDeltaEvent)
    ) == "model answer"
    assert isinstance(events[-1], ResearchCompleteEvent)
    assert run.status == ResearchRunStatus.COMPLETED
    assert run.answer == "model answer"
    assert web.closed is True


@pytest.mark.asyncio
async def test_first_progress_event_precedes_blocked_provider(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class BlockingWeb(FakeWeb):
        async def search(self, objective, queries):
            entered.set()
            await asyncio.Event().wait()

    web = BlockingWeb()
    service = build_service(tmp_path / "research.sqlite3", FakeLLM(), web)
    await service.start()
    run_id = await service.create_run("question")
    stream = service.stream_run(run_id, "question")

    first = await asyncio.wait_for(anext(stream), timeout=1)

    assert isinstance(first, ResearchProgressEvent)
    assert first.stage == "started"
    await stream.aclose()
    run = await service.get_run(run_id)
    assert run.status == ResearchRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_total_provider_failure_emits_error_without_answer(tmp_path: Path) -> None:
    service = build_service(
        tmp_path / "research.sqlite3",
        FakeLLM(),
        FakeWeb(error=RuntimeError("offline")),
    )
    await service.start()
    run_id = await service.create_run("question")

    events = [event async for event in service.stream_run(run_id, "question")]
    run = await service.get_run(run_id)

    assert isinstance(events[-1], ResearchErrorEvent)
    assert not any(isinstance(event, ResearchAnswerDeltaEvent) for event in events)
    assert run.status == ResearchRunStatus.FAILED
    assert run.answer is None


@pytest.mark.asyncio
async def test_cancelling_stream_persists_cancelled_status(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class BlockingLLM(FakeLLM):
        async def stream(self, system_prompt, user_prompt):
            entered.set()
            await asyncio.Event().wait()
            if False:
                yield TextChunk(text="", usage=TokenUsage())

    web = FakeWeb()
    service = build_service(tmp_path / "research.sqlite3", BlockingLLM(), web)
    await service.start()
    run_id = await service.create_run("question")
    stream = service.stream_run(run_id, "question")

    while True:
        event = await anext(stream)
        if isinstance(event, ResearchProgressEvent) and event.stage == "drafting":
            break
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    run = await service.get_run(run_id)
    assert run.status == ResearchRunStatus.CANCELLED
    assert web.closed is True
