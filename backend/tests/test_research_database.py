from pathlib import Path

import pytest

from database import DocumentDatabase, ResearchDatabase
from models import (
    DocumentRecord,
    DocumentStatus,
    EvidenceItem,
    ResearchCounters,
    ResearchRunStatus,
    ResearchStopReason,
    TokenUsage,
    TraceEventStatus,
)


def document_record(tmp_path: Path, document_id: str, status: DocumentStatus) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        content_hash=f"hash-{document_id}",
        version=1,
        filename=f"{document_id}.txt",
        content_type="text/plain",
        size=1,
        status=status,
        original_path=tmp_path / f"{document_id}.txt",
        parsed_path=tmp_path / f"{document_id}.json",
        embedding_model="embedding-model",
        error=None,
        created_at=f"2026-01-0{1 if document_id == 'ready' else 2}T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_ready_document_snapshot_excludes_partially_indexed_rows(tmp_path: Path) -> None:
    database = DocumentDatabase(tmp_path / "documents.sqlite3")
    await database.initialize()
    await database.register_batch(
        [
            document_record(tmp_path, "ready", DocumentStatus.READY),
            document_record(tmp_path, "processing", DocumentStatus.PROCESSING),
        ],
        [],
        "embedding-model",
    )

    ready = await database.list_ready()

    assert [item.document_id for item in ready] == ["ready"]


@pytest.mark.asyncio
async def test_run_evidence_trace_and_monotonic_transitions(tmp_path: Path) -> None:
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    await database.initialize()
    await database.create_run("run-1", "question")

    assert await database.mark_running("run-1") is True
    evidence = EvidenceItem(
        evidence_id="e1",
        source_id="D1",
        source_type="document",
        source_key="chunk-1",
        title="notes.txt",
        document_id="doc-1",
        passage="evidence",
    )
    assert await database.add_run_evidence("run-1", evidence) is True
    assert await database.add_run_evidence("run-1", evidence) is False
    await database.accept_evidence("run-1", ["e1"])
    event = await database.add_trace_event(
        "run-1",
        "assess",
        TraceEventStatus.COMPLETED,
        duration_ms=12,
        result_summary="gap: current source date",
        source_ids=["D1"],
        token_usage=TokenUsage(total_tokens=7),
    )
    finished = await database.finish_run(
        "run-1",
        ResearchRunStatus.COMPLETED,
        ResearchStopReason.SUFFICIENT,
        answer="answer [D1]",
        counters=ResearchCounters(evidence_accepted=1),
        token_usage=TokenUsage(total_tokens=7),
    )

    run = await database.get_run("run-1")
    assert finished is True
    assert run.status == ResearchRunStatus.COMPLETED
    assert run.evidence[0].accepted is True
    assert run.trace[0].event_id == event.event_id
    assert run.trace[0].duration_ms == 12
    assert await database.mark_running("run-1") is False
    assert not await database.finish_run(
        "run-1", ResearchRunStatus.FAILED, ResearchStopReason.ERROR
    )


@pytest.mark.asyncio
async def test_startup_interrupts_stale_runs(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite3"
    database = ResearchDatabase(path)
    await database.initialize()
    await database.create_run("queued", "question")
    await database.create_run("running", "question")
    await database.mark_running("running")

    recreated = ResearchDatabase(path)
    await recreated.initialize()
    await recreated.mark_unfinished_interrupted()

    for run_id in ("queued", "running"):
        run = await recreated.get_run(run_id)
        assert run.status == ResearchRunStatus.INTERRUPTED
        assert run.stop_reason == ResearchStopReason.ERROR
        assert run.trace[-1].step == "run"
