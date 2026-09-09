from types import SimpleNamespace

import pytest
from qdrant_client import models as qdrant

from models import SparseVector
from vector_store import QdrantVectorStore


class FakeQdrantClient:
    def __init__(self) -> None:
        self.query_kwargs = None

    async def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "document_id": "ready-1",
                        "chunk_id": "chunk-1",
                        "filename": "notes.txt",
                        "page": 3,
                        "text": "useful",
                    }
                ),
                SimpleNamespace(
                    payload={
                        "document_id": "ready-1",
                        "chunk_id": "chunk-1",
                        "filename": "notes.txt",
                        "text": "duplicate",
                    }
                ),
                SimpleNamespace(
                    payload={
                        "document_id": "not-ready",
                        "chunk_id": "chunk-2",
                        "filename": "partial.txt",
                        "text": "must be filtered",
                    }
                ),
                SimpleNamespace(payload={"chunk_id": "invalid"}),
            ]
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_hybrid_query_builds_two_prefetches_and_rrf() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        url="http://unused",
        collection="documents",
        dense_dimensions=3,
        client=client,
    )

    results = await store.hybrid_search(
        [0.1, 0.2, 0.3],
        SparseVector(indices=[1, 9], values=[0.5, 0.7]),
        ["ready-1"],
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    query = client.query_kwargs
    assert query["limit"] == 12
    assert len(query["prefetch"]) == 2
    assert [item.using for item in query["prefetch"]] == ["dense", "sparse"]
    assert all(item.limit == 20 for item in query["prefetch"])
    assert query["query"] == qdrant.FusionQuery(fusion=qdrant.Fusion.RRF)
    for prefetch in query["prefetch"]:
        assert prefetch.filter.must[0].match.any == ["ready-1"]


@pytest.mark.asyncio
async def test_hybrid_query_returns_before_qdrant_when_no_ready_documents() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore("http://unused", "documents", 3, client=client)

    assert await store.hybrid_search(
        [0.1, 0.2, 0.3], SparseVector(indices=[], values=[]), []
    ) == []
    assert client.query_kwargs is None
