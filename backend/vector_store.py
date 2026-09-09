from collections.abc import Sequence
from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant

from models import DocumentChunk, DocumentRecord, RetrievedChunk, SparseVector

UPSERT_BATCH_SIZE = 64


class VectorStore(Protocol):
    async def initialize(self) -> None: ...

    async def add_document(
        self,
        document: DocumentRecord,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> None: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        ready_document_ids: Sequence[str],
        *,
        prefetch_limit: int = 20,
        limit: int = 12,
    ) -> list[RetrievedChunk]: ...

    async def close(self) -> None: ...


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        dense_dimensions: int,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self.dense_dimensions = dense_dimensions
        self.client = client or AsyncQdrantClient(url=url, timeout=30)

    async def initialize(self) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": qdrant.VectorParams(
                        size=self.dense_dimensions,
                        distance=qdrant.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": qdrant.SparseVectorParams(modifier=qdrant.Modifier.IDF)
                },
            )
            return

        collection = await self.client.get_collection(self.collection)
        vectors = collection.config.params.vectors
        sparse_vectors = collection.config.params.sparse_vectors or {}

        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        sparse = sparse_vectors.get("sparse")
        sparse_modifier = str(getattr(sparse, "modifier", "")).lower()

        if (
            dense is None
            or dense.size != self.dense_dimensions
            or dense.distance != qdrant.Distance.COSINE
        ):
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' has an incompatible dense vector."
            )
        if sparse is None or not sparse_modifier.endswith("idf"):
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' has an incompatible sparse vector."
            )

    async def add_document(
        self,
        document: DocumentRecord,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> None:
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("Every chunk must have one dense and one sparse vector.")

        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            if len(dense) != self.dense_dimensions:
                raise ValueError(
                    f"Dense vector has {len(dense)} values; expected {self.dense_dimensions}."
                )
            if len(sparse.indices) != len(sparse.values):
                raise ValueError("Sparse vector indices and values must have the same length.")

            points.append(
                qdrant.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        "dense": dense,
                        "sparse": qdrant.SparseVector(
                            indices=sparse.indices,
                            values=sparse.values,
                        ),
                    },
                    payload={
                        "document_id": document.document_id,
                        "version": document.version,
                        "chunk_id": chunk.chunk_id,
                        "filename": document.filename,
                        "page": chunk.page,
                        "section": chunk.section,
                        "text": chunk.text,
                        "parent_section_id": chunk.parent_section_id,
                    },
                )
            )

        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + UPSERT_BATCH_SIZE],
                wait=True,
            )

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=qdrant.FilterSelector(
                filter=qdrant.Filter(
                    must=[
                        qdrant.FieldCondition(
                            key="document_id",
                            match=qdrant.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        ready_document_ids: Sequence[str],
        *,
        prefetch_limit: int = 20,
        limit: int = 12,
    ) -> list[RetrievedChunk]:
        if not ready_document_ids:
            return []
        if len(dense_vector) != self.dense_dimensions:
            raise ValueError(
                f"Dense vector has {len(dense_vector)} values; expected {self.dense_dimensions}."
            )
        if len(sparse_vector.indices) != len(sparse_vector.values):
            raise ValueError("Sparse vector indices and values must have the same length.")
        prefetch_limit = min(max(1, prefetch_limit), 20)
        limit = min(max(1, limit), 12)

        ready_filter = qdrant.Filter(
            must=[
                qdrant.FieldCondition(
                    key="document_id",
                    match=qdrant.MatchAny(any=list(ready_document_ids)),
                )
            ]
        )
        response = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qdrant.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=ready_filter,
                    limit=prefetch_limit,
                ),
                qdrant.Prefetch(
                    query=qdrant.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    filter=ready_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=qdrant.FusionQuery(fusion=qdrant.Fusion.RRF),
            query_filter=ready_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for point in response.points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or "")
            document_id = str(payload.get("document_id") or "")
            filename = str(payload.get("filename") or "")
            text = str(payload.get("text") or "").strip()
            if (
                not chunk_id
                or chunk_id in seen
                or document_id not in ready_document_ids
                or not filename
                or not text
            ):
                continue
            seen.add(chunk_id)
            page = payload.get("page")
            chunks.append(
                RetrievedChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    filename=filename,
                    page=int(page) if page is not None else None,
                    section=(str(payload["section"]) if payload.get("section") else None),
                    text=text,
                )
            )
            if len(chunks) >= limit:
                break
        return chunks

    async def close(self) -> None:
        await self.client.close()
