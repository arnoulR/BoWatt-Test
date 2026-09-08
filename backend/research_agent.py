from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's request directly.
You do not have research or browsing tools, so do not claim to have researched sources or invent
citations."""


class StreamingChatModel(Protocol):
    def astream(self, input: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]: ...


class ResearchAgent:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-5.6-luna",
        chat_model: StreamingChatModel | None = None,
    ) -> None:
        if chat_model is not None:
            self._chat_model = chat_model
            return

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")

        self._chat_model = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            reasoning_effort="none",
        )

    async def stream(self, request: str) -> AsyncIterator[str]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=request),
        ]

        async for chunk in self._chat_model.astream(messages):
            if chunk.text:
                yield str(chunk.text)
