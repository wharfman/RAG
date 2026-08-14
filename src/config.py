from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    database_url: str
    data_dir: Path
    generation_model: str
    embedding_model: str
    embedding_dimensions: int
    retrieval_count: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        data_dir = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))).resolve()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://rag:rag@localhost:5432/rag",
            ),
            data_dir=data_dir,
            generation_model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-5.6-terra"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
            retrieval_count=int(os.getenv("RAG_TOP_K", "6")),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to the repository root .env file."
            )
        return self.openai_api_key
