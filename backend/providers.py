import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

from models import (
    FetchedPage,
    SearchResult,
    SparseVector,
    TokenUsage,
)


class ProviderError(RuntimeError):
    def __init__(self, provider: str, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.retryable = retryable


@dataclass(slots=True)
class TextChunk:
    text: str
    usage: TokenUsage


@dataclass(slots=True)
class FetchResult:
    pages: list[FetchedPage]
    errors: list[str]


class LLMProvider(Protocol):
    model_name: str

    def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[TextChunk]: ...


class WebSearchProvider(Protocol):
    async def search(self, objective: str, queries: Sequence[str]) -> list[SearchResult]: ...

    async def fetch(self, urls: Sequence[str], objective: str) -> FetchResult: ...

    async def close(self) -> None: ...


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class SparseEmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def embed_query(self, text: str) -> SparseVector: ...


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

    async def embed_query(self, text: str) -> SparseVector:
        return await asyncio.to_thread(self._embed_query, text)

    def _embed(self, texts: list[str]) -> list[SparseVector]:
        vectors = self._get_model().embed(texts)
        return [
            SparseVector(
                indices=vector.indices.tolist(),
                values=vector.values.tolist(),
            )
            for vector in vectors
        ]

    def _embed_query(self, text: str) -> SparseVector:
        model = self._get_model()
        vector = next(iter(model.query_embed(text)))
        return SparseVector(
            indices=vector.indices.tolist(),
            values=vector.values.tolist(),
        )

    def _get_model(self):
        from fastembed import SparseTextEmbedding

        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SparseTextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
        return self._model


class OpenAILLMProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        reasoning_effort: str,
        max_retries: int,
        timeout_seconds: float,
        semaphore: asyncio.Semaphore | None = None,
        chat_model=None,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._semaphore = semaphore or asyncio.Semaphore(1_000_000)
        self._chat_model = chat_model or ChatOpenAI(
            api_key=api_key,
            model=model_name,
            reasoning_effort=reasoning_effort,
            use_responses_api=True,
            store=False,
            stream_usage=True,
            max_retries=max_retries,
            timeout=timeout_seconds,
        )

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[TextChunk]:
        try:
            async with self._semaphore:
                async with asyncio.timeout(self.timeout_seconds):
                    async for chunk in self._chat_model.astream(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt),
                        ]
                    ):
                        yield TextChunk(
                            text=str(chunk.text or ""),
                            usage=_usage_from_message(chunk),
                        )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise normalize_provider_error("openai", error) from error


class ParallelWebProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_retries: int,
        timeout_seconds: float,
        semaphore: asyncio.Semaphore | None = None,
        client=None,
    ) -> None:
        if client is None:
            from parallel import AsyncParallel

            client = AsyncParallel(
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout_seconds,
            )
        self._client = client
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._semaphore = semaphore or asyncio.Semaphore(1_000_000)
        self._session_id: str | None = None

    async def search(self, objective: str, queries: Sequence[str]) -> list[SearchResult]:
        clean_queries = deduplicate_queries(queries)
        if not clean_queries:
            return []
        try:
            async with self._semaphore:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._client.search(
                        objective=objective,
                        search_queries=clean_queries,
                        session_id=self._session_id,
                        client_model=self._model_name,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise normalize_provider_error("parallel", error) from error

        response_session = getattr(response, "session_id", None)
        if response_session:
            self._session_id = str(response_session)
        seen: set[str] = set()
        results: list[SearchResult] = []
        for item in getattr(response, "results", []) or []:
            canonical = canonicalize_url(str(getattr(item, "url", "")))
            excerpts = [
                str(value).strip()
                for value in (getattr(item, "excerpts", []) or [])
                if str(value).strip()
            ]
            if not canonical or canonical in seen or not excerpts:
                continue
            seen.add(canonical)
            results.append(
                SearchResult(
                    url=str(getattr(item, "url", canonical)),
                    canonical_url=canonical,
                    title=str(getattr(item, "title", "") or ""),
                    publish_date=_optional_string(getattr(item, "publish_date", None)),
                    excerpts=excerpts,
                )
            )
        return results

    async def fetch(self, urls: Sequence[str], objective: str) -> FetchResult:
        clean_urls = list(dict.fromkeys(filter(None, (canonicalize_url(url) for url in urls))))
        if not clean_urls:
            return FetchResult(pages=[], errors=[])
        try:
            async with self._semaphore:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._client.extract(
                        urls=clean_urls,
                        objective=objective,
                        session_id=self._session_id,
                        client_model=self._model_name,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise normalize_provider_error("parallel", error) from error

        response_session = getattr(response, "session_id", None)
        if response_session:
            self._session_id = str(response_session)
        pages: list[FetchedPage] = []
        for item in getattr(response, "results", []) or []:
            canonical = canonicalize_url(str(getattr(item, "url", "")))
            passages = [
                str(value).strip()
                for value in (getattr(item, "excerpts", []) or [])
                if str(value).strip()
            ]
            if not canonical or not passages:
                continue
            pages.append(
                FetchedPage(
                    url=str(getattr(item, "url", canonical)),
                    canonical_url=canonical,
                    title=str(getattr(item, "title", "") or ""),
                    publish_date=_optional_string(getattr(item, "publish_date", None)),
                    passages=passages,
                )
            )
        errors = [
            f"{getattr(item, 'error_type', 'fetch_error')}: "
            f"{str(getattr(item, 'content', 'Unable to fetch URL.'))[:300]}"
            for item in (getattr(response, "errors", []) or [])
        ]
        return FetchResult(pages=pages, errors=errors)

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def deduplicate_queries(queries: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for query in queries:
        clean = " ".join(str(query).split())
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def canonicalize_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname.lower()
    try:
        port = parts.port
    except ValueError:
        return ""
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    if parts.username or parts.password:
        return ""
    path = parts.path or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def normalize_provider_error(provider: str, error: Exception) -> ProviderError:
    status_code = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    retryable = (
        status_code in {408, 409, 429}
        or (isinstance(status_code, int) and status_code >= 500)
        or "timeout" in name
        or "connection" in name
    )
    code = str(status_code) if status_code is not None else type(error).__name__
    message = str(error).strip() or "Provider request failed."
    return ProviderError(provider, code, message[:500], retryable=retryable)


def _usage_from_message(message) -> TokenUsage:
    raw = getattr(message, "usage_metadata", None) or {}
    input_details = raw.get("input_token_details") or {}
    output_details = raw.get("output_token_details") or {}
    return TokenUsage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        total_tokens=int(raw.get("total_tokens") or 0),
        reasoning_tokens=int(output_details.get("reasoning") or 0),
        cache_read_tokens=int(input_details.get("cache_read") or 0),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
