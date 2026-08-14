from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from src.config import Settings
from src.interface.base_datastore import BaseDatastore, DataItem, SearchResult


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(768), nullable=False)
    heading: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


class Datastore(BaseDatastore):
    """PostgreSQL + pgvector storage and cosine-similarity retrieval."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        if self.settings.embedding_dimensions != 1536:
            raise ValueError("The current pgvector schema requires 1536-dimensional embeddings")
        self.engine = create_engine(
            self.settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.require_openai_key())
        return self._client

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )

    def reset(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.initialize()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_vector(self, content: str) -> list[float]:
        return self.get_vectors([content])[0]

    def get_vectors(self, contents: Sequence[str]) -> list[list[float]]:
        if not contents:
            return []
        response = self.client.embeddings.create(
            input=list(contents),
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def add_document(
        self,
        path: Path,
        file_hash: str,
        items: Sequence[DataItem],
        batch_size: int = 64,
    ) -> int:
        embeddings: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            embeddings.extend(
                self.get_vectors([item.content for item in items[start : start + batch_size]])
            )

        with Session(self.engine) as session, session.begin():
            existing = session.scalar(
                select(DocumentRecord).where(DocumentRecord.file_name == path.name)
            )
            if existing:
                session.delete(existing)
                session.flush()
            document = DocumentRecord(
                file_name=path.name,
                sha256=file_hash,
                byte_size=path.stat().st_size,
                chunk_count=len(items),
                indexed_at=datetime.now(UTC),
            )
            session.add(document)
            session.flush()
            session.add_all(
                ChunkRecord(
                    document_id=document.id,
                    chunk_index=item.chunk_index,
                    source=item.source,
                    heading=item.heading,
                    content=item.content,
                    content_hash=item.content_hash,
                    embedding=embedding,
                )
                for item, embedding in zip(items, embeddings, strict=True)
            )
        return len(items)

    def add_items(self, items: list[DataItem]) -> None:
        raise NotImplementedError("Use add_document() so document hashes remain consistent")

    def remove_missing_documents(self, file_names: Iterable[str]) -> int:
        names = set(file_names)
        with Session(self.engine) as session, session.begin():
            stale = list(
                session.scalars(
                    select(DocumentRecord).where(DocumentRecord.file_name.not_in(names))
                )
            )
            for document in stale:
                session.delete(document)
            return len(stale)

    def document_map(self) -> dict[str, dict[str, object]]:
        with Session(self.engine) as session:
            records = session.scalars(select(DocumentRecord).order_by(DocumentRecord.file_name))
            return {
                record.file_name: {
                    "sha256": record.sha256,
                    "byte_size": record.byte_size,
                    "chunk_count": record.chunk_count,
                    "indexed_at": record.indexed_at,
                }
                for record in records
            }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.get_vector(query)
        distance = ChunkRecord.embedding.cosine_distance(query_vector)
        statement = (
            select(ChunkRecord, DocumentRecord.file_name, distance.label("distance"))
            .join(DocumentRecord)
            .order_by(distance)
            .limit(top_k)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).all()
        return [
            SearchResult(
                content=chunk.content,
                source=chunk.source,
                file_name=file_name,
                heading=chunk.heading,
                score=max(0.0, 1.0 - float(distance_value)),
            )
            for chunk, file_name, distance_value in rows
        ]

    def stats(self) -> dict[str, int]:
        with Session(self.engine) as session:
            return {
                "documents": session.scalar(select(func.count(DocumentRecord.id))) or 0,
                "chunks": session.scalar(select(func.count(ChunkRecord.id))) or 0,
            }
