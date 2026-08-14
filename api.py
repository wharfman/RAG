from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Settings
from src.impl import Datastore, Indexer, ResponseGenerator, Retriever
from src.ingestion import sync_data_directory
from src.rag_pipeline import RAGPipeline

settings = Settings.from_env()
datastore = Datastore(settings)
indexer = Indexer()
pipeline = RAGPipeline(
    datastore=datastore,
    indexer=indexer,
    retriever=Retriever(datastore),
    response_generator=ResponseGenerator(settings),
)
ingestion_lock = Lock()

app = FastAPI(title="RAG Docs API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


@app.get("/api/health")
def health() -> dict[str, object]:
    database_available = datastore.ping()
    stats = {"documents": 0, "chunks": 0}
    if database_available:
        try:
            stats = datastore.stats()
        except Exception:
            database_available = False
    return {
        "status": "ok" if database_available else "degraded",
        "database": database_available,
        "openai_key_configured": bool(settings.openai_api_key),
        **stats,
    }


@app.get("/api/documents")
def documents() -> dict[str, object]:
    paths = indexer.discover(settings.data_dir)
    records: dict[str, dict[str, object]] = {}
    database_available = datastore.ping()
    if database_available:
        try:
            records = datastore.document_map()
        except Exception:
            database_available = False

    files = []
    for path in paths:
        file_hash = indexer.file_hash(path)
        record = records.get(path.name)
        files.append(
            {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "indexed": bool(record and record["sha256"] == file_hash),
                "chunks": record["chunk_count"] if record else 0,
                "indexed_at": record["indexed_at"] if record else None,
            }
        )
    return {"database": database_available, "files": files}


@app.post("/api/ingest")
def ingest() -> dict[str, object]:
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion run is already in progress")
    try:
        report = sync_data_directory(datastore, indexer, settings.data_dir)
        return {
            "indexed_files": report.indexed_files,
            "skipped_files": report.skipped_files,
            "removed_files": report.removed_files,
            "chunks_added": report.chunks_added,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        ingestion_lock.release()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    try:
        if not datastore.ping():
            raise HTTPException(status_code=503, detail="PostgreSQL is not available")
        stats = datastore.stats()
        if not stats["chunks"]:
            raise HTTPException(
                status_code=409,
                detail="No documents are indexed. Use the Sync documents button first.",
            )
        history = [message.model_dump() for message in request.history]
        answer, results = pipeline.answer_query(request.question.strip(), history)
        return {
            "answer": answer,
            "sources": [
                {
                    "source": result.source,
                    "file_name": result.file_name,
                    "heading": result.heading,
                    "score": round(result.score, 4),
                }
                for result in results
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
