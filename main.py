from __future__ import annotations

import argparse
import json

from src.config import Settings
from src.impl import Datastore, Indexer, ResponseGenerator, Retriever
from src.ingestion import sync_data_directory, validate_data_directory
from src.rag_pipeline import RAGPipeline


def create_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or Settings.from_env()
    datastore = Datastore(settings)
    return RAGPipeline(
        datastore=datastore,
        indexer=Indexer(),
        retriever=Retriever(datastore),
        response_generator=ResponseGenerator(settings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Synchronize data/ into PostgreSQL/pgvector")
    subparsers.add_parser("validate", help="Validate and chunk every supported data file")
    subparsers.add_parser("reset", help="Reset the pgvector tables")
    query = subparsers.add_parser("query", help="Ask a question about indexed files")
    query.add_argument("prompt")
    args = parser.parse_args()

    settings = Settings.from_env()
    pipeline = create_pipeline(settings)
    if args.command == "validate":
        print(json.dumps(validate_data_directory(pipeline.indexer, settings.data_dir), indent=2))
    elif args.command == "ingest":
        report = sync_data_directory(pipeline.datastore, pipeline.indexer, settings.data_dir)
        print(json.dumps(report.__dict__, indent=2))
    elif args.command == "reset":
        pipeline.reset()
        print("PostgreSQL/pgvector tables reset.")
    elif args.command == "query":
        print(pipeline.process_query(args.prompt))


if __name__ == "__main__":
    main()
