from src.config import Settings


def test_default_models(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    settings = Settings.from_env()
    assert settings.generation_model == "gpt-5.6-terra"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
