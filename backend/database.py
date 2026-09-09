import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from models import (
    Citation,
    DocumentRecord,
    DocumentStatus,
    EvidenceItem,
    EvidenceSourceType,
    ReadyDocument,
    ResearchCounters,
    ResearchRunResponse,
    ResearchRunStatus,
    ResearchStopReason,
    TokenUsage,
    TraceEvent,
    TraceEventStatus,
)


class DocumentDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA busy_timeout=5000")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    version INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    parsed_path TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status)"
            )
            await connection.commit()

    async def mark_unfinished_interrupted(self) -> None:
        now = _now()
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                UPDATE documents
                SET status = ?, error = ?, updated_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    DocumentStatus.INTERRUPTED,
                    "Backend stopped before processing finished.",
                    now,
                    DocumentStatus.QUEUED,
                    DocumentStatus.PROCESSING,
                ),
            )
            await connection.commit()

    async def get_by_hashes(self, content_hashes: Sequence[str]) -> dict[str, DocumentRecord]:
        if not content_hashes:
            return {}

        placeholders = ",".join("?" for _ in content_hashes)
        query = f"SELECT * FROM documents WHERE content_hash IN ({placeholders})"  # noqa: S608

        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(query, tuple(content_hashes))
            rows = await cursor.fetchall()

        return {row["content_hash"]: _to_document(row) for row in rows}

    async def register_batch(
        self,
        new_documents: Sequence[DocumentRecord],
        retry_document_ids: Sequence[str],
        embedding_model: str,
    ) -> None:
        now = _now()
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("PRAGMA busy_timeout=5000")
            await connection.execute("BEGIN IMMEDIATE")

            for document in new_documents:
                await connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, content_hash, version, filename, content_type, size,
                        status, original_path, parsed_path, embedding_model, error,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        document.content_hash,
                        document.version,
                        document.filename,
                        document.content_type,
                        document.size,
                        document.status,
                        str(document.original_path),
                        str(document.parsed_path),
                        document.embedding_model,
                        document.error,
                        document.created_at,
                        document.updated_at,
                    ),
                )

            for document_id in retry_document_ids:
                await connection.execute(
                    """
                    UPDATE documents
                    SET status = ?, embedding_model = ?, error = NULL, updated_at = ?
                    WHERE document_id = ?
                    """,
                    (DocumentStatus.QUEUED, embedding_model, now, document_id),
                )

            await connection.commit()

    async def get(self, document_id: str) -> DocumentRecord | None:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = await cursor.fetchone()

        return _to_document(row) if row else None

    async def get_ready_embedding_models(self) -> set[str]:
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                "SELECT DISTINCT embedding_model FROM documents WHERE status = ?",
                (DocumentStatus.READY,),
            )
            rows = await cursor.fetchall()
        return {row[0] for row in rows}

    async def list_ready(self) -> list[ReadyDocument]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT document_id, filename, original_path, created_at
                FROM documents
                WHERE status = ?
                ORDER BY created_at, document_id
                """,
                (DocumentStatus.READY,),
            )
            rows = await cursor.fetchall()

        return [
            ReadyDocument(
                document_id=row["document_id"],
                filename=row["filename"],
                original_path=Path(row["original_path"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def set_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error: str | None = None,
    ) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                UPDATE documents
                SET status = ?, error = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (status, error, _now(), document_id),
            )
            await connection.commit()


class ResearchDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA busy_timeout=5000")
            await connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    counters_json TEXT NOT NULL DEFAULT '{}',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    stop_reason TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS research_runs_status_idx
                ON research_runs(status);

                CREATE TABLE IF NOT EXISTS research_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    document_id TEXT,
                    url TEXT,
                    publish_date TEXT,
                    page INTEGER,
                    section TEXT,
                    passage TEXT NOT NULL,
                    summary TEXT,
                    round INTEGER NOT NULL,
                    accepted INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, source_key, content_hash),
                    FOREIGN KEY(run_id) REFERENCES research_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS research_evidence_run_idx
                ON research_evidence(run_id, round, source_id);

                CREATE TABLE IF NOT EXISTS trace_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    duration_ms INTEGER,
                    provider TEXT,
                    model TEXT,
                    input_summary TEXT,
                    result_summary TEXT,
                    source_ids_json TEXT NOT NULL DEFAULT '[]',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    FOREIGN KEY(run_id) REFERENCES research_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS trace_events_run_idx
                ON trace_events(run_id, event_id);
                """
            )
            await connection.commit()

    async def create_run(self, run_id: str, question: str) -> None:
        now = _now()
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                INSERT INTO research_runs (
                    run_id, question, status, counters_json, token_usage_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    question,
                    ResearchRunStatus.QUEUED,
                    ResearchCounters().model_dump_json(),
                    TokenUsage().model_dump_json(),
                    now,
                    now,
                ),
            )
            await connection.commit()

    async def mark_running(self, run_id: str) -> bool:
        now = _now()
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                """
                UPDATE research_runs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE run_id = ? AND status = ?
                """,
                (
                    ResearchRunStatus.RUNNING,
                    now,
                    now,
                    run_id,
                    ResearchRunStatus.QUEUED,
                ),
            )
            await connection.commit()
            return cursor.rowcount == 1

    async def finish_run(
        self,
        run_id: str,
        status: ResearchRunStatus,
        stop_reason: ResearchStopReason,
        *,
        answer: str | None = None,
        citations: Sequence[Citation] = (),
        counters: ResearchCounters | None = None,
        token_usage: TokenUsage | None = None,
        error: str | None = None,
    ) -> bool:
        if status not in {
            ResearchRunStatus.COMPLETED,
            ResearchRunStatus.FAILED,
            ResearchRunStatus.CANCELLED,
            ResearchRunStatus.INTERRUPTED,
        }:
            raise ValueError(f"{status} is not a terminal research status.")

        now = _now()
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                """
                UPDATE research_runs
                SET status = ?, answer = ?, citations_json = ?, counters_json = ?,
                    token_usage_json = ?, stop_reason = ?, error = ?, updated_at = ?,
                    completed_at = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (
                    status,
                    answer,
                    json.dumps([item.model_dump(mode="json") for item in citations]),
                    (counters or ResearchCounters()).model_dump_json(),
                    (token_usage or TokenUsage()).model_dump_json(),
                    stop_reason,
                    error,
                    now,
                    now,
                    run_id,
                    ResearchRunStatus.QUEUED,
                    ResearchRunStatus.RUNNING,
                ),
            )
            await connection.commit()
            return cursor.rowcount == 1

    async def update_progress(
        self,
        run_id: str,
        counters: ResearchCounters,
        token_usage: TokenUsage,
    ) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                UPDATE research_runs
                SET counters_json = ?, token_usage_json = ?, updated_at = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (
                    counters.model_dump_json(),
                    token_usage.model_dump_json(),
                    _now(),
                    run_id,
                    ResearchRunStatus.QUEUED,
                    ResearchRunStatus.RUNNING,
                ),
            )
            await connection.commit()

    async def mark_unfinished_interrupted(self) -> None:
        now = _now()
        message = "Backend stopped before research finished."
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                "SELECT run_id FROM research_runs WHERE status IN (?, ?)",
                (ResearchRunStatus.QUEUED, ResearchRunStatus.RUNNING),
            )
            rows = await cursor.fetchall()
            if not rows:
                return

            await connection.execute(
                """
                UPDATE research_runs
                SET status = ?, stop_reason = ?, error = ?, updated_at = ?, completed_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    ResearchRunStatus.INTERRUPTED,
                    ResearchStopReason.ERROR,
                    message,
                    now,
                    now,
                    ResearchRunStatus.QUEUED,
                    ResearchRunStatus.RUNNING,
                ),
            )
            for row in rows:
                await connection.execute(
                    """
                    INSERT INTO trace_events (
                        run_id, step, status, created_at, result_summary, error
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["run_id"],
                        "run",
                        TraceEventStatus.FAILED,
                        now,
                        "Research interrupted during backend restart.",
                        message,
                    ),
                )
            await connection.commit()

    async def add_run_evidence(self, run_id: str, evidence: EvidenceItem) -> bool:
        content_hash = hashlib.sha256(evidence.passage.encode("utf-8")).hexdigest()
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO research_evidence (
                    evidence_id, run_id, source_id, source_type, source_key, title,
                    document_id, url, page, section, passage, summary, round,
                    publish_date, accepted, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    run_id,
                    evidence.source_id,
                    evidence.source_type,
                    evidence.source_key,
                    evidence.title,
                    evidence.document_id,
                    evidence.url,
                    evidence.page,
                    evidence.section,
                    evidence.passage,
                    evidence.summary,
                    evidence.round,
                    evidence.publish_date,
                    int(evidence.accepted),
                    content_hash,
                    _now(),
                ),
            )
            await connection.commit()
            return cursor.rowcount == 1

    async def accept_evidence(self, run_id: str, evidence_ids: Sequence[str]) -> None:
        if not evidence_ids:
            return
        placeholders = ",".join("?" for _ in evidence_ids)
        query = (  # noqa: S608
            f"UPDATE research_evidence SET accepted = 1 "
            f"WHERE run_id = ? AND evidence_id IN ({placeholders})"
        )
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(query, (run_id, *evidence_ids))
            await connection.commit()

    async def add_trace_event(
        self,
        run_id: str,
        step: str,
        status: TraceEventStatus,
        *,
        duration_ms: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        input_summary: str | None = None,
        result_summary: str | None = None,
        source_ids: Sequence[str] = (),
        token_usage: TokenUsage | None = None,
        error: str | None = None,
    ) -> TraceEvent:
        created_at = _now()
        usage = token_usage or TokenUsage()
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                """
                INSERT INTO trace_events (
                    run_id, step, status, created_at, duration_ms, provider, model,
                    input_summary, result_summary, source_ids_json, token_usage_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    status,
                    created_at,
                    duration_ms,
                    provider,
                    model,
                    input_summary,
                    result_summary,
                    json.dumps(list(source_ids)),
                    usage.model_dump_json(),
                    error,
                ),
            )
            await connection.commit()
            event_id = int(cursor.lastrowid)

        return TraceEvent(
            event_id=event_id,
            run_id=run_id,
            step=step,
            status=status,
            created_at=created_at,
            duration_ms=duration_ms,
            provider=provider,
            model=model,
            input_summary=input_summary,
            result_summary=result_summary,
            source_ids=list(source_ids),
            token_usage=usage,
            error=error,
        )

    async def get_status(self, run_id: str) -> ResearchRunStatus | None:
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                "SELECT status FROM research_runs WHERE run_id = ?", (run_id,)
            )
            row = await cursor.fetchone()
        return ResearchRunStatus(row[0]) if row else None

    async def get_run(self, run_id: str) -> ResearchRunResponse | None:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
            )
            row = await cursor.fetchone()
        if row is None:
            return None

        evidence = await self.get_evidence(run_id)
        trace = await self.get_trace_events(run_id)
        return ResearchRunResponse(
            run_id=row["run_id"],
            question=row["question"],
            status=ResearchRunStatus(row["status"]),
            answer=row["answer"],
            citations=[Citation.model_validate(item) for item in _json(row["citations_json"], [])],
            evidence=evidence,
            stop_reason=(
                ResearchStopReason(row["stop_reason"]) if row["stop_reason"] else None
            ),
            error=row["error"],
            counters=ResearchCounters.model_validate(_json(row["counters_json"], {})),
            token_usage=TokenUsage.model_validate(_json(row["token_usage_json"], {})),
            created_at=row["created_at"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            trace=trace,
        )

    async def get_evidence(self, run_id: str) -> list[EvidenceItem]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT * FROM research_evidence
                WHERE run_id = ?
                ORDER BY round, source_id, created_at, evidence_id
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [_to_evidence(row) for row in rows]

    async def get_trace_events(self, run_id: str, after_id: int = 0) -> list[TraceEvent]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT * FROM trace_events
                WHERE run_id = ? AND event_id > ?
                ORDER BY event_id
                """,
                (run_id, after_id),
            )
            rows = await cursor.fetchall()
        return [_to_trace_event(row) for row in rows]


def _to_document(row: aiosqlite.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=row["document_id"],
        content_hash=row["content_hash"],
        version=row["version"],
        filename=row["filename"],
        content_type=row["content_type"],
        size=row["size"],
        status=DocumentStatus(row["status"]),
        original_path=Path(row["original_path"]),
        parsed_path=Path(row["parsed_path"]),
        embedding_model=row["embedding_model"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: str | None, default: object) -> object:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _to_evidence(row: aiosqlite.Row) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=row["evidence_id"],
        source_id=row["source_id"],
        source_type=EvidenceSourceType(row["source_type"]),
        source_key=row["source_key"],
        title=row["title"],
        document_id=row["document_id"],
        url=row["url"],
        publish_date=row["publish_date"],
        page=row["page"],
        section=row["section"],
        passage=row["passage"],
        summary=row["summary"],
        round=row["round"],
        accepted=bool(row["accepted"]),
    )


def _to_trace_event(row: aiosqlite.Row) -> TraceEvent:
    return TraceEvent(
        event_id=row["event_id"],
        run_id=row["run_id"],
        step=row["step"],
        status=TraceEventStatus(row["status"]),
        created_at=row["created_at"],
        duration_ms=row["duration_ms"],
        provider=row["provider"],
        model=row["model"],
        input_summary=row["input_summary"],
        result_summary=row["result_summary"],
        source_ids=list(_json(row["source_ids_json"], [])),
        token_usage=TokenUsage.model_validate(_json(row["token_usage_json"], {})),
        error=row["error"],
    )
