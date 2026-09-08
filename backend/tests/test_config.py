from config import Settings


def test_chat_model_defaults_to_luna(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "gpt-5.6-luna"


def test_chat_model_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "test-chat-model"
