from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import Settings
from ingestion import IngestionService, StorageUnavailable, UploadValidationError
from models import (
    ResearchAnswerDeltaEvent,
    ResearchCompleteEvent,
    ResearchErrorEvent,
    ResearchProgressEvent,
    ResearchRequest,
    ResearchRunResponse,
    UploadResponse,
)
from research_service import ResearchService
from services import build_ingestion_service, build_research_service


def create_app(
    settings: Settings | None = None,
    ingestion_service: IngestionService | None = None,
    research_service: ResearchService | None = None,
) -> FastAPI:
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        ingestion = ingestion_service or build_ingestion_service(app_settings)
        research: ResearchService | None = None
        try:
            await ingestion.start()
            research = research_service or build_research_service(app_settings, ingestion)
            await research.start()
            app.state.ingestion = ingestion
            app.state.research = research
            yield
        finally:
            if research is not None:
                await research.stop()
            await ingestion.stop()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Research-Run-ID"],
    )

    @app.post(
        "/api/sources",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_sources(
        files: Annotated[list[UploadFile], File(description="Source documents")],
    ) -> UploadResponse:
        try:
            return await app.state.ingestion.register(files)
        except UploadValidationError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        except StorageUnavailable as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document storage is unavailable.",
            ) from error

    @app.post("/api/research")
    async def research(payload: ResearchRequest) -> StreamingResponse:
        run_id = await app.state.research.create_run(payload.request)

        async def markdown_stream() -> AsyncGenerator[str, None]:
            answer_parts: list[str] = []
            async for event in app.state.research.stream_run(run_id, payload.request):
                if isinstance(event, ResearchProgressEvent):
                    yield f"{event.message}\n"
                elif isinstance(event, ResearchAnswerDeltaEvent):
                    answer_parts.append(event.delta)
                elif isinstance(event, ResearchCompleteEvent):
                    yield "Complete.\n\nAnswer:\n\n"
                    yield "".join(answer_parts)
                elif isinstance(event, ResearchErrorEvent):
                    yield f"Failed: {event.message}\n"

        return StreamingResponse(
            markdown_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Research-Run-ID": run_id,
            },
        )

    @app.get("/api/research/{run_id}", response_model=ResearchRunResponse)
    async def get_research(run_id: str) -> ResearchRunResponse:
        run = await app.state.research.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return run

    return app


app = create_app()
