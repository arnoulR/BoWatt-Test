import asyncio
import os

from config import Settings
from database import DocumentDatabase, ResearchDatabase
from ingestion import IngestionService
from parser import DoclingParser
from providers import (
    FastEmbedSparseProvider,
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
    ParallelWebProvider,
)
from research_agent import HybridResearchRetriever, ResearchLimits
from research_service import ResearchService
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


def build_research_service(
    settings: Settings,
    ingestion: IngestionService,
) -> ResearchService:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    if not settings.parallel_api_key:
        raise RuntimeError("PARALLEL_API_KEY is required.")

    semaphore = asyncio.Semaphore(settings.research_max_concurrent_calls)
    limits = ResearchLimits(
        max_fetched_urls=settings.research_max_fetched_urls,
    )
    llm = OpenAILLMProvider(
        api_key=settings.openai_api_key,
        model_name=settings.openai_chat_model,
        reasoning_effort=settings.openai_reasoning_effort,
        max_retries=settings.research_max_retries,
        timeout_seconds=settings.research_provider_timeout_seconds,
        semaphore=semaphore,
    )
    retriever = HybridResearchRetriever(
        embeddings=ingestion.embeddings,
        sparse_embeddings=ingestion.sparse_embeddings,
        vector_store=ingestion.vector_store,
        semaphore=semaphore,
        timeout_seconds=settings.research_provider_timeout_seconds,
    )

    def build_web_provider() -> ParallelWebProvider:
        return ParallelWebProvider(
            api_key=settings.parallel_api_key or "",
            model_name=settings.openai_chat_model,
            max_retries=settings.research_max_retries,
            timeout_seconds=settings.research_provider_timeout_seconds,
            semaphore=semaphore,
        )

    return ResearchService(
        document_database=ingestion.database,
        research_database=ResearchDatabase(settings.database_path),
        llm=llm,
        retriever=retriever,
        web_factory=build_web_provider,
        limits=limits,
    )
