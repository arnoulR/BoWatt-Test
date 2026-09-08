import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from models import SparseVector


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class SparseEmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            max_retries=2,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embeddings.aembed_documents(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)


class FastEmbedSparseProvider:
    def __init__(self, model: str, cache_dir: Path) -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self._model = None

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await asyncio.to_thread(self._embed, list(texts))

    def _embed(self, texts: list[str]) -> list[SparseVector]:
        from fastembed import SparseTextEmbedding

        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SparseTextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )

        vectors = self._model.embed(texts)
        return [
            SparseVector(
                indices=vector.indices.tolist(),
                values=vector.values.tolist(),
            )
            for vector in vectors
        ]
