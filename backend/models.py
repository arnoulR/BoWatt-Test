from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ResearchRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ResearchStopReason(StrEnum):
    SUFFICIENT = "sufficient"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    ERROR = "error"


class EvidenceSourceType(StrEnum):
    DOCUMENT = "document"
    WEB = "web"


class TraceEventStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)
    model_config = ConfigDict(str_strip_whitespace=True)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0

    def plus(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


class ResearchCounters(BaseModel):
    web_queries_used: int = 0
    urls_fetched: int = 0
    provider_calls: int = 0
    evidence_candidates: int = 0
    evidence_accepted: int = 0


class ReadyDocument(BaseModel):
    document_id: str
    filename: str
    original_path: Path
    created_at: str


class SearchResult(BaseModel):
    url: str
    canonical_url: str
    title: str = ""
    publish_date: str | None = None
    excerpts: list[str] = Field(default_factory=list)


class FetchedPage(BaseModel):
    url: str
    canonical_url: str
    title: str = ""
    publish_date: str | None = None
    passages: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    source_id: str
    source_type: EvidenceSourceType
    source_key: str
    title: str
    passage: str
    summary: str | None = None
    round: int = 1
    accepted: bool = False
    document_id: str | None = None
    url: str | None = None
    publish_date: str | None = None
    page: int | None = None
    section: str | None = None


class Citation(BaseModel):
    marker: str
    source_id: str
    source_type: EvidenceSourceType
    title: str
    document_id: str | None = None
    url: str | None = None
    page: int | None = None


class TraceEvent(BaseModel):
    event_id: int
    run_id: str
    step: str
    status: TraceEventStatus
    created_at: str
    duration_ms: int | None = None
    provider: str | None = None
    model: str | None = None
    input_summary: str | None = None
    result_summary: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class ResearchRunResponse(BaseModel):
    run_id: str
    question: str
    status: ResearchRunStatus
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    stop_reason: ResearchStopReason | None = None
    error: str | None = None
    counters: ResearchCounters = Field(default_factory=ResearchCounters)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None
    trace: list[TraceEvent] = Field(default_factory=list)


class ResearchProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    stage: Literal[
        "started", "searching", "reading_sources", "sources_found", "drafting"
    ]
    message: str


class ResearchAnswerDeltaEvent(BaseModel):
    type: Literal["answer_delta"] = "answer_delta"
    delta: str


class ResearchCompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    run_id: str
    citations: list[Citation] = Field(default_factory=list)


class ResearchErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


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


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    text: str
