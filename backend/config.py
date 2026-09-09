from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str = "low"
    openai_embedding_model: str = "text-embedding-3-small"
    parallel_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "shared_documents"
    data_dir: Path = Path("data")
    max_upload_files: int = 10
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:5173"

    embedding_dimensions: int = 1536
    chunk_tokens: int = 600
    sparse_model: str = "Qdrant/bm25"

    research_max_fetched_urls: int = Field(default=8, ge=0)
    research_max_concurrent_calls: int = Field(default=3, ge=1)
    research_max_retries: int = Field(default=2, ge=0)
    research_provider_timeout_seconds: float = Field(default=30, gt=0)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "documents.sqlite3"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
