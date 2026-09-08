from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from models import DocumentRecord, DocumentStatus


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
