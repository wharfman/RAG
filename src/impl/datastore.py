from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
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
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    search_vector: Mapped[object] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(heading, '') || ' ' || coalesce(content, ''))",
            persisted=True,
        ),
        nullable=False,
    )
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


@dataclass(frozen=True)
class RankedResult:
    chunk_id: int
    result: SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedResult]],
    top_k: int,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse independent rankings using RRF; raw channel scores are never blended."""
    if top_k < 1:
        return []
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    results: dict[int, SearchResult] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, ranked in enumerate(ranking, start=1):
            if ranked.chunk_id in seen:
                continue
            seen.add(ranked.chunk_id)
            scores[ranked.chunk_id] = scores.get(ranked.chunk_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            best_rank[ranked.chunk_id] = min(best_rank.get(ranked.chunk_id, rank), rank)
            results[ranked.chunk_id] = ranked.result

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id),
    )[:top_k]
    return [
        results[chunk_id].model_copy(update={"score": scores[chunk_id]})
        for chunk_id in ordered_ids
    ]


BM25_SEARCH_SQL = text(
    """
    WITH query_terms AS (
        SELECT DISTINCT term.lexeme
        FROM unnest(to_tsvector('english', CAST(:query AS text)))
             AS term(lexeme, positions, weights)
    ),
    query_filter AS (
        SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS value
        FROM query_terms
    ),
    corpus_lengths AS (
        SELECT
            c.id,
            COALESCE(SUM(cardinality(term.positions)), 0)::double precision AS doc_length
        FROM chunks AS c
        LEFT JOIN LATERAL unnest(c.search_vector) AS term(lexeme, positions, weights)
            ON TRUE
        GROUP BY c.id
    ),
    corpus AS (
        SELECT
            COUNT(*)::double precision AS document_count,
            COALESCE(AVG(doc_length), 1.0)::double precision AS avg_doc_length
        FROM corpus_lengths
    ),
    matching_terms AS (
        SELECT
            c.id AS chunk_id,
            qt.lexeme,
            cardinality(term.positions)::double precision AS term_frequency
        FROM chunks AS c
        CROSS JOIN query_filter AS qf
        CROSS JOIN LATERAL unnest(c.search_vector) AS term(lexeme, positions, weights)
        JOIN query_terms AS qt ON qt.lexeme = term.lexeme
        WHERE qf.value IS NOT NULL AND c.search_vector @@ qf.value
    ),
    document_frequency AS (
        SELECT lexeme, COUNT(*)::double precision AS document_frequency
        FROM matching_terms
        GROUP BY lexeme
    ),
    bm25 AS (
        SELECT
            mt.chunk_id,
            SUM(
                ln(
                    1.0 + (corpus.document_count - df.document_frequency + 0.5)
                    / (df.document_frequency + 0.5)
                )
                * (
                    mt.term_frequency * (:k1 + 1.0)
                    / (
                        mt.term_frequency
                        + :k1 * (
                            1.0 - :b
                            + :b * lengths.doc_length / NULLIF(corpus.avg_doc_length, 0.0)
                        )
                    )
                )
            ) AS score
        FROM matching_terms AS mt
        JOIN document_frequency AS df USING (lexeme)
        JOIN corpus_lengths AS lengths ON lengths.id = mt.chunk_id
        CROSS JOIN corpus
        GROUP BY mt.chunk_id
    )
    SELECT
        c.id,
        c.content,
        c.source,
        d.file_name,
        c.heading,
        bm25.score
    FROM bm25
    JOIN chunks AS c ON c.id = bm25.chunk_id
    JOIN documents AS d ON d.id = c.document_id
    ORDER BY bm25.score DESC, c.id ASC
    LIMIT :candidate_limit
    """
)


class Datastore(BaseDatastore):
    """Hybrid pgvector and PostgreSQL BM25 retrieval fused with RRF."""

    RRF_RANK_CONSTANT = 60
    CANDIDATE_MULTIPLIER = 4
    MIN_CANDIDATES = 20
    BM25_K1 = 1.2
    BM25_B = 0.75

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
            # create_all() does not migrate an existing chunks table.
            connection.execute(
                text(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector "
                    "GENERATED ALWAYS AS ("
                    "to_tsvector('english', coalesce(heading, '') || ' ' || "
                    "coalesce(content, ''))) STORED"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_search_vector_gin "
                    "ON chunks USING gin (search_vector)"
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
        if top_k < 1 or not query.strip():
            return []
        query_vector = self.get_vector(query)
        candidate_limit = max(self.MIN_CANDIDATES, top_k * self.CANDIDATE_MULTIPLIER)
        semantic_results = self._semantic_search(query_vector, candidate_limit)
        lexical_results = self._bm25_search(query, candidate_limit)
        return reciprocal_rank_fusion(
            [semantic_results, lexical_results],
            top_k=top_k,
            rank_constant=self.RRF_RANK_CONSTANT,
        )

    def _semantic_search(
        self, query_vector: list[float], candidate_limit: int
    ) -> list[RankedResult]:
        distance = ChunkRecord.embedding.cosine_distance(query_vector)
        statement = (
            select(
                ChunkRecord.id,
                ChunkRecord.content,
                ChunkRecord.source,
                DocumentRecord.file_name,
                ChunkRecord.heading,
                distance.label("distance"),
            )
            .join(DocumentRecord)
            .order_by(distance, ChunkRecord.id)
            .limit(candidate_limit)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).all()
        return [
            RankedResult(
                chunk_id=chunk_id,
                result=SearchResult(
                    content=content,
                    source=source,
                    file_name=file_name,
                    heading=heading,
                    score=max(0.0, 1.0 - float(distance_value)),
                ),
            )
            for chunk_id, content, source, file_name, heading, distance_value in rows
        ]

    def _bm25_search(self, query: str, candidate_limit: int) -> list[RankedResult]:
        with Session(self.engine) as session:
            rows = session.execute(
                BM25_SEARCH_SQL,
                {
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "k1": self.BM25_K1,
                    "b": self.BM25_B,
                },
            ).mappings()
            return [
                RankedResult(
                    chunk_id=row["id"],
                    result=SearchResult(
                        content=row["content"],
                        source=row["source"],
                        file_name=row["file_name"],
                        heading=row["heading"],
                        score=float(row["score"]),
                    ),
                )
                for row in rows
            ]

    def stats(self) -> dict[str, int]:
        with Session(self.engine) as session:
            return {
                "documents": session.scalar(select(func.count(DocumentRecord.id))) or 0,
                "chunks": session.scalar(select(func.count(ChunkRecord.id))) or 0,
            }
