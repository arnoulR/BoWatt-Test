import asyncio
import hashlib
import shutil
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, status

from config import Settings
from database import DocumentDatabase
from models import (
    DocumentRecord,
    DocumentStatus,
    StagedUpload,
    UploadedDocument,
    UploadResponse,
)
from parser import DocumentParser
from providers import EmbeddingProvider, SparseEmbeddingProvider
from vector_store import VectorStore

# Add more extensions here when the frontend supports them.
SUPPORTED_SUFFIXES = {".txt"}


class UploadValidationError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT):
        super().__init__(message)
        self.status_code = status_code


class StorageUnavailable(Exception):
    pass


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        database: DocumentDatabase,
        parser: DocumentParser,
        embeddings: EmbeddingProvider,
        sparse_embeddings: SparseEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.database = database
        self.parser = parser
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.vector_store = vector_store
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.register_lock = asyncio.Lock()
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._create_directories()
        await self.database.initialize()
        await self.database.mark_unfinished_interrupted()
        indexed_models = await self.database.get_ready_embedding_models()
        if indexed_models - {self.settings.openai_embedding_model}:
            raise RuntimeError(
                "The embedding model changed. Re-index documents in a new collection "
                "and data directory."
            )
        await self.vector_store.initialize()
        self.worker_task = asyncio.create_task(self._worker(), name="document-ingestion")

    async def stop(self) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
        await self.parser.close()
        await self.vector_store.close()

    async def register(self, files: Sequence[UploadFile]) -> UploadResponse:
        self._validate_files(files)
        staged: list[StagedUpload] = []

        try:
            for file in files:
                staged.append(await self._stage(file))
            return await self._register_staged(staged)
        except UploadValidationError:
            raise
        except (OSError, RuntimeError, sqlite3.Error) as error:
            raise StorageUnavailable from error
        finally:
            for upload in staged:
                upload.path.unlink(missing_ok=True)

    def _validate_files(self, files: Sequence[UploadFile]) -> None:
        if not files:
            raise UploadValidationError("At least one file is required.")
        if len(files) > self.settings.max_upload_files:
            raise UploadValidationError(
                f"A maximum of {self.settings.max_upload_files} files can be uploaded at once."
            )

        for file in files:
            name = Path(file.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise UploadValidationError(
                    f"Unsupported file type: {name or 'unnamed file'}.",
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                )

    async def _stage(self, file: UploadFile) -> StagedUpload:
        name = Path(file.filename or "source").name
        suffix = Path(name).suffix.lower()
        stage_path = self.settings.staging_dir / f"{uuid4()}{suffix}"
        content_hash = hashlib.sha256()
        size = 0

        try:
            with stage_path.open("xb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise UploadValidationError(
                            f"{name} is larger than {self.settings.max_upload_mb} MiB.",
                            status.HTTP_413_CONTENT_TOO_LARGE,
                        )
                    content_hash.update(chunk)
                    target.write(chunk)
        except Exception:
            stage_path.unlink(missing_ok=True)
            raise

        if size == 0:
            stage_path.unlink(missing_ok=True)
            raise UploadValidationError(f"{name} is empty.")

        return StagedUpload(
            name=name,
            size=size,
            content_type=file.content_type or "application/octet-stream",
            content_hash=content_hash.hexdigest(),
            suffix=suffix,
            path=stage_path,
        )

    async def _register_staged(self, staged: list[StagedUpload]) -> UploadResponse:
        async with self.register_lock:
            known = await self.database.get_by_hashes([item.content_hash for item in staged])
            new_documents: list[DocumentRecord] = []
            retry_ids: set[str] = set()
            queue_ids: set[str] = set()
            created_dirs: list[Path] = []
            restored_files: list[Path] = []
            uploaded: list[UploadedDocument] = []

            try:
                for item in staged:
                    document = known.get(item.content_hash)
                    duplicate = document is not None

                    if document is None:
                        document = self._new_document(item)
                        document.original_path.parent.mkdir(parents=True)
                        item.path.replace(document.original_path)
                        created_dirs.append(document.original_path.parent)
                        new_documents.append(document)
                        known[item.content_hash] = document
                        queue_ids.add(document.document_id)
                    else:
                        if not document.original_path.exists():
                            document.original_path.parent.mkdir(parents=True, exist_ok=True)
                            item.path.replace(document.original_path)
                            restored_files.append(document.original_path)

                        if document.status in {
                            DocumentStatus.FAILED,
                            DocumentStatus.INTERRUPTED,
                        }:
                            document.status = DocumentStatus.QUEUED
                            retry_ids.add(document.document_id)
                            queue_ids.add(document.document_id)

                    uploaded.append(
                        UploadedDocument(
                            name=item.name,
                            size=item.size,
                            type=item.content_type,
                            document_id=document.document_id,
                            status=document.status,
                            duplicate=duplicate,
                        )
                    )

                await self.database.register_batch(
                    new_documents,
                    sorted(retry_ids),
                    self.settings.openai_embedding_model,
                )
            except Exception:
                for directory in created_dirs:
                    shutil.rmtree(directory, ignore_errors=True)
                for restored_file in restored_files:
                    restored_file.unlink(missing_ok=True)
                raise

            for document_id in sorted(queue_ids):
                self.queue.put_nowait(document_id)

            return UploadResponse(uploaded=uploaded)

    def _new_document(self, upload: StagedUpload) -> DocumentRecord:
        document_id = str(uuid4())
        document_dir = self.settings.documents_dir / document_id
        now = datetime.now(UTC).isoformat()
        return DocumentRecord(
            document_id=document_id,
            content_hash=upload.content_hash,
            version=1,
            filename=upload.name,
            content_type=upload.content_type,
            size=upload.size,
            status=DocumentStatus.QUEUED,
            original_path=document_dir / f"original{upload.suffix}",
            parsed_path=document_dir / "parsed.json",
            embedding_model=self.settings.openai_embedding_model,
            error=None,
            created_at=now,
            updated_at=now,
        )

    async def _worker(self) -> None:
        while True:
            document_id = await self.queue.get()
            try:
                await self._process(document_id)
            except asyncio.CancelledError:
                await self.database.set_status(document_id, DocumentStatus.INTERRUPTED)
                await self._remove_vectors(document_id)
                raise
            except Exception as error:
                await self._remove_vectors(document_id)
                message = f"{type(error).__name__}: {error}"[:500]
                await self.database.set_status(document_id, DocumentStatus.FAILED, message)
            finally:
                self.queue.task_done()

    async def _process(self, document_id: str) -> None:
        document = await self.database.get(document_id)
        if document is None or document.status != DocumentStatus.QUEUED:
            return

        await self.database.set_status(document_id, DocumentStatus.PROCESSING)
        await self.vector_store.delete_document(document_id)
        chunks = await self.parser.parse(
            document.original_path,
            document.parsed_path,
            document.document_id,
            document.version,
        )
        if not chunks:
            raise ValueError("The document did not contain readable text.")

        texts = [chunk.embedding_text for chunk in chunks]
        dense_vectors, sparse_vectors = await asyncio.gather(
            self.embeddings.embed_documents(texts),
            self.sparse_embeddings.embed_documents(texts),
        )
        await self.vector_store.add_document(document, chunks, dense_vectors, sparse_vectors)
        await self.database.set_status(document_id, DocumentStatus.READY)

    async def _remove_vectors(self, document_id: str) -> None:
        try:
            await self.vector_store.delete_document(document_id)
        except Exception:
            # Cleanup is best-effort. Readiness in SQLite remains the retrieval gate.
            pass

    def _create_directories(self) -> None:
        self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
        self.settings.staging_dir.mkdir(parents=True, exist_ok=True)
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
