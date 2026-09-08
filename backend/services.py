import os

from config import Settings
from database import DocumentDatabase
from ingestion import IngestionService
from parser import DoclingParser
from providers import FastEmbedSparseProvider, OpenAIEmbeddingProvider
from vector_store import QdrantVectorStore


def build_ingestion_service(settings: Settings) -> IngestionService:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(settings.model_cache_dir / "huggingface"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.model_cache_dir / "tiktoken"))
    os.environ.setdefault("TORCH_HOME", str(settings.model_cache_dir / "torch"))

    return IngestionService(
        settings=settings,
        database=DocumentDatabase(settings.database_path),
        parser=DoclingParser(
            embedding_model=settings.openai_embedding_model,
            chunk_tokens=settings.chunk_tokens,
            model_cache_dir=settings.model_cache_dir,
        ),
        embeddings=OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        ),
        sparse_embeddings=FastEmbedSparseProvider(
            model=settings.sparse_model,
            cache_dir=settings.model_cache_dir / "fastembed",
        ),
        vector_store=QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dense_dimensions=settings.embedding_dimensions,
        ),
    )
