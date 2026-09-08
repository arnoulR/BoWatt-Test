from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)
    model_config = ConfigDict(str_strip_whitespace=True)


class UploadedDocument(BaseModel):
    name: str
    size: int
    type: str
    document_id: str
    status: DocumentStatus
    duplicate: bool


class UploadResponse(BaseModel):
    uploaded: list[UploadedDocument]


@dataclass(slots=True)
class StagedUpload:
    name: str
    size: int
    content_type: str
    content_hash: str
    suffix: str
    path: Path


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    content_hash: str
    version: int
    filename: str
    content_type: str
    size: int
    status: DocumentStatus
    original_path: Path
    parsed_path: Path
    embedding_model: str
    error: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    page: int | None
    section: str | None
    text: str
    parent_section_id: str | None
    embedding_text: str


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]
