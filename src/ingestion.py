from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.impl.datastore import Datastore
from src.impl.indexer import Indexer


@dataclass
class IngestionReport:
    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    removed_files: int = 0
    chunks_added: int = 0


def sync_data_directory(datastore: Datastore, indexer: Indexer, data_dir: Path) -> IngestionReport:
    datastore.initialize()
    paths = indexer.discover(data_dir)
    if not paths:
        raise RuntimeError(f"No supported documents found under {data_dir}")

    existing = datastore.document_map()
    report = IngestionReport()
    for path in paths:
        file_hash = indexer.file_hash(path)
        record = existing.get(path.name)
        if record and record["sha256"] == file_hash:
            report.skipped_files.append(path.name)
            continue
        items = indexer.index_file(path)
        report.chunks_added += datastore.add_document(path, file_hash, items)
        report.indexed_files.append(path.name)

    report.removed_files = datastore.remove_missing_documents(path.name for path in paths)
    return report


def validate_data_directory(indexer: Indexer, data_dir: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for path in indexer.discover(data_dir):
        items = indexer.index_file(path)
        reports.append(
            {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "chunks": len(items),
                "min_chunk_chars": min(len(item.content) for item in items),
                "max_chunk_chars": max(len(item.content) for item in items),
                "empty_chunks": sum(not item.content.strip() for item in items),
                "unique_sources": len({item.source for item in items}),
            }
        )
    return reports
