from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from research_agent import ResearchAgent


class FakeChatModel:
    def __init__(self) -> None:
        self.messages = []

    async def astream(self, messages) -> AsyncIterator[AIMessageChunk]:
        self.messages = messages
        yield AIMessageChunk(content="Hello")
        yield AIMessageChunk(content="")
        yield AIMessageChunk(content=" world")


@pytest.mark.asyncio
async def test_agent_streams_chat_model_text_in_order() -> None:
    model = FakeChatModel()
    agent = ResearchAgent(chat_model=model)

    chunks = [chunk async for chunk in agent.stream("Say hello")]

    assert chunks == ["Hello", " world"]
    assert isinstance(model.messages[0], SystemMessage)
    assert isinstance(model.messages[1], HumanMessage)
    assert model.messages[1].content == "Say hello"


def test_agent_requires_api_key_without_injected_model() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        ResearchAgent()
