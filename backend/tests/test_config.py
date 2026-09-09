from config import Settings


def test_chat_model_defaults_to_luna(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "gpt-5.6-luna"


def test_chat_model_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "test-chat-model"


def test_research_defaults_and_embedding_model_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "test-chat-model"
    assert settings.openai_embedding_model == "test-embedding-model"
    assert settings.openai_reasoning_effort == "low"
    assert settings.research_max_fetched_urls == 8
    assert settings.research_max_concurrent_calls == 3
    assert settings.research_max_retries == 2
