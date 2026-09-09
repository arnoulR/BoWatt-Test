import asyncio
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from database import DocumentDatabase, ResearchDatabase
from models import (
    ResearchAnswerDeltaEvent,
    ResearchCompleteEvent,
    ResearchErrorEvent,
    ResearchProgressEvent,
    ResearchRunResponse,
    ResearchRunStatus,
    ResearchStopReason,
    TraceEventStatus,
)
from providers import LLMProvider, ProviderError, WebSearchProvider
from research_agent import (
    AgentCompleteEvent,
    AnswerDeltaEvent,
    ProgressEvent,
    ResearchAgent,
    ResearchLimits,
    ResearchRetriever,
)

type ResearchStreamEvent = (
    ResearchProgressEvent
    | ResearchAnswerDeltaEvent
    | ResearchCompleteEvent
    | ResearchErrorEvent
)


class ResearchService:
    def __init__(
        self,
        document_database: DocumentDatabase,
        research_database: ResearchDatabase,
        llm: LLMProvider,
        retriever: ResearchRetriever,
        web_factory: Callable[[], WebSearchProvider],
        limits: ResearchLimits | None = None,
    ) -> None:
        self.document_database = document_database
        self.database = research_database
        self._llm = llm
        self._retriever = retriever
        self._web_factory = web_factory
        self._limits = limits or ResearchLimits()
        self._stopping = False

    async def start(self) -> None:
        await self.database.initialize()
        await self.database.mark_unfinished_interrupted()

    async def stop(self) -> None:
        self._stopping = True

    async def create_run(self, question: str) -> str:
        run_id = str(uuid4())
        await self.database.create_run(run_id, question)
        return run_id

    async def stream_run(
        self, run_id: str, question: str
    ) -> AsyncIterator[ResearchStreamEvent]:
        if not await self.database.mark_running(run_id):
            yield ResearchErrorEvent(message="Research run could not be started.")
            return

        web = self._web_factory()
        answer_parts: list[str] = []
        completed = False
        try:
            started_message = "Research started."
            await self.database.add_trace_event(
                run_id,
                "started",
                TraceEventStatus.STARTED,
                result_summary=started_message,
            )
            yield ResearchProgressEvent(
                stage="started",
                message=started_message,
            )
            documents = await self.document_database.list_ready()
            agent = ResearchAgent(
                llm=self._llm,
                retriever=self._retriever,
                web=web,
                limits=self._limits,
            )
            async for event in agent.stream(run_id, question, documents):
                if isinstance(event, ProgressEvent):
                    await self.database.add_trace_event(
                        run_id,
                        event.stage,
                        TraceEventStatus.STARTED,
                        result_summary=event.message,
                    )
                    yield ResearchProgressEvent(
                        stage=event.stage,
                        message=event.message,
                    )
                    continue

                if isinstance(event, AnswerDeltaEvent):
                    answer_parts.append(event.delta)
                    yield ResearchAnswerDeltaEvent(delta=event.delta)
                    continue

                if isinstance(event, AgentCompleteEvent):
                    for evidence in event.evidence:
                        await self.database.add_run_evidence(run_id, evidence)
                    await self.database.finish_run(
                        run_id,
                        ResearchRunStatus.COMPLETED,
                        ResearchStopReason.SUFFICIENT,
                        answer=event.answer,
                        citations=event.citations,
                        counters=event.counters,
                        token_usage=event.token_usage,
                    )
                    await self.database.add_trace_event(
                        run_id,
                        "run",
                        TraceEventStatus.COMPLETED,
                        provider="openai",
                        model=self._llm.model_name,
                        result_summary="Research completed.",
                        source_ids=[item.source_id for item in event.citations],
                        token_usage=event.token_usage,
                    )
                    completed = True
                    yield ResearchCompleteEvent(
                        run_id=run_id,
                        citations=event.citations,
                    )
        except asyncio.CancelledError:
            interrupted = self._stopping
            status = (
                ResearchRunStatus.INTERRUPTED if interrupted else ResearchRunStatus.CANCELLED
            )
            reason = ResearchStopReason.ERROR if interrupted else ResearchStopReason.CANCELLED
            message = (
                "Backend stopped before research finished."
                if interrupted
                else "Research was cancelled."
            )
            await self.database.finish_run(
                run_id,
                status,
                reason,
                answer="".join(answer_parts) or None,
                error=message,
            )
            await self.database.add_trace_event(
                run_id,
                "run",
                TraceEventStatus.FAILED if interrupted else TraceEventStatus.CANCELLED,
                result_summary=message,
                error=message,
            )
            raise
        except Exception as error:
            message = _concise_error(error)
            await self.database.finish_run(
                run_id,
                ResearchRunStatus.FAILED,
                ResearchStopReason.ERROR,
                answer="".join(answer_parts) or None,
                error=message,
            )
            await self.database.add_trace_event(
                run_id,
                "run",
                TraceEventStatus.FAILED,
                result_summary="Research failed.",
                error=message,
            )
            yield ResearchErrorEvent(message=message)
        finally:
            if not completed:
                status = await self.database.get_status(run_id)
                if status == ResearchRunStatus.RUNNING:
                    message = "Research stream was closed before completion."
                    await self.database.finish_run(
                        run_id,
                        ResearchRunStatus.CANCELLED,
                        ResearchStopReason.CANCELLED,
                        answer="".join(answer_parts) or None,
                        error=message,
                    )
                    await self.database.add_trace_event(
                        run_id,
                        "run",
                        TraceEventStatus.CANCELLED,
                        result_summary=message,
                        error=message,
                    )
            await _close_web(web)

    async def get_run(self, run_id: str) -> ResearchRunResponse | None:
        return await self.database.get_run(run_id)


async def _close_web(web: WebSearchProvider) -> None:
    try:
        await web.close()
    except Exception:
        pass


def _concise_error(error: Exception) -> str:
    if isinstance(error, ProviderError):
        return f"{error.provider} {error.code}: {str(error)}"[:500]
    return f"{type(error).__name__}: {error}"[:500]
