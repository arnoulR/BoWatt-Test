from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from config import Settings
from main import create_app


class FakeIngestionService:
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeResearchAgent:
    def __init__(self) -> None:
        self.requests: list[str] = []

    async def stream(self, request: str) -> AsyncIterator[str]:
        self.requests.append(request)
        yield "first "
        yield "second"


def test_research_endpoint_streams_plain_text() -> None:
    agent = FakeResearchAgent()
    app = create_app(
        settings=Settings(_env_file=None),
        ingestion_service=FakeIngestionService(),
        research_agent=agent,
    )

    with TestClient(app) as client:
        response = client.post("/api/research", json={"request": "Test request"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "first second"
    assert agent.requests == ["Test request"]


def test_research_endpoint_rejects_blank_requests() -> None:
    app = create_app(
        settings=Settings(_env_file=None),
        ingestion_service=FakeIngestionService(),
        research_agent=FakeResearchAgent(),
    )

    with TestClient(app) as client:
        response = client.post("/api/research", json={"request": "   "})

    assert response.status_code == 422
