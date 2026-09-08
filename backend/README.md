# Backend

FastAPI backend for source uploads and the research-agent frontend.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
cd backend
cp .env.example .env
uv sync
```

Add your `OPENAI_API_KEY` to `.env` and start Qdrant:

```sh
docker compose up -d qdrant
uv run uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
```

To run the backend and Qdrant together:

```sh
docker compose up --build
```

The first document may take longer because Docling and FastEmbed download their local models.

`OPENAI_CHAT_MODEL` selects the model used by the research endpoint and defaults to
`gpt-5.6-luna`.

## Upload sources

`POST /api/sources` accepts TXT files under the repeated field `files`.

```sh
curl -X POST http://localhost:8787/api/sources \
  -F "files=@notes.txt"
```

The endpoint returns once files are queued. Documents move through `queued`, `processing`,
`ready`, `failed`, or `interrupted` in SQLite. Reupload an interrupted or failed file to retry it.
Changing `OPENAI_EMBEDDING_MODEL` requires a new collection and data directory.

## Research requests

`POST /api/research` accepts JSON containing a non-empty `request` and streams the model's
plain-text response:

```sh
curl -N http://localhost:8787/api/research \
  -H "Content-Type: application/json" \
  -d '{"request":"Explain why the sky is blue."}'
```

This first version is a stateless LangChain chat call. It does not use uploaded documents,
search the web, retain conversation history, or perform multi-step research.

## Tests

```sh
# Unit tests
uv run pytest

# API, ingestion, and persistence tests
uv run pytest integration_tests

# Code checks
uv run ruff check .
```

The real Qdrant test is skipped unless `QDRANT_TEST_URL` points to a running server.
