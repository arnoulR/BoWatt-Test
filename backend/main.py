from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)
    model_config = ConfigDict(str_strip_whitespace=True)

async def generate_answer(request: str):
    yield f"# Research\n\nYour question: {request}\n\n"
    await asyncio.sleep(0.5)
    yield "LLM answer here\n\n"
    

@app.post("/api/research")
def research(payload: ResearchRequest):
    return StreamingResponse(
        generate_answer(payload.request),
        media_type="text/plain",
    )
