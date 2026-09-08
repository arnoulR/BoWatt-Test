import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import Settings
from ingestion import IngestionService, StorageUnavailable, UploadValidationError
from models import ResearchRequest, UploadResponse
from services import build_ingestion_service


async def generate_answer(request: str) -> AsyncIterator[str]:
    yield f"# Research\n\nYour question: {request}\n\n"
    await asyncio.sleep(0.5)
    yield "LLM answer here\n\n"


def create_app(
    settings: Settings | None = None,
    ingestion_service: IngestionService | None = None,
) -> FastAPI:
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        service = ingestion_service or build_ingestion_service(app_settings)
        await service.start()
        app.state.ingestion = service

        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
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
    def research(payload: ResearchRequest) -> StreamingResponse:
        return StreamingResponse(
            generate_answer(payload.request),
            media_type="text/plain",
        )

    return app


app = create_app()
