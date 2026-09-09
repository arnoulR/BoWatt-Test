# Backend

FastAPI backend for source uploads and the research-agent frontend.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
cd backend
cp .env.example .env
uv sync
```

Add your `OPENAI_API_KEY` and `PARALLEL_API_KEY` to `.env`, then start Qdrant:

```sh
docker compose up -d qdrant
uv run uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
```

To run the backend and Qdrant together:

```sh
docker compose up --build
```

The first document may take longer because Docling and FastEmbed download their local models.

`OPENAI_CHAT_MODEL` selects the research model and defaults to `gpt-5.6-luna` with low
reasoning effort. It is independent from `OPENAI_EMBEDDING_MODEL`, so changing the chat model
does not require re-indexing documents. The Compose service uses Qdrant 1.19.0 to match the
locked Python client.

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

`POST /api/research` accepts JSON containing a non-empty `request` and streams plain-text
progress followed by the completed Markdown answer, matching the frontend's byte-streaming
contract. The
`X-Research-Run-ID` response header identifies the persisted run:

```sh
curl -N http://localhost:8787/api/research \
  -H "Content-Type: application/json" \
  -d '{"request":"Explain why the sky is blue."}'
```

The agent performs one bounded research pass. It searches ready uploaded TXT files and the web,
then writes a model-generated Markdown answer using the collected context. The response reports
when research starts, searches documents and the web, reads selected web sources, has sources
ready (including document and web counts), drafts, and completes. The completed answer follows
those stages. If research fails, the response ends with a readable `Failed` stage and does not
show an incomplete answer.

Inspect the persisted result with:

```text
GET  /api/research/{run_id}
```

Closing the response stream cancels the active run. Runs are bounded by fetched URLs, outbound
concurrency, retries, and provider timeouts; each limit is configurable through the `RESEARCH_*`
variables in `.env.example`.

## Tests

```sh
# Unit tests
uv run pytest tests

# API, ingestion, and persistence tests
uv run pytest integration_tests

# Code checks
uv run ruff check .
```
