from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from src.interface import (
    BaseDatastore,
    BaseEvaluator,
    BaseIndexer,
    BaseResponseGenerator,
    BaseRetriever,
    EvaluationResult,
    SearchResult,
)


@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""

    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: BaseEvaluator | None = None

    def reset(self) -> None:
        """Reset the datastore"""
        self.datastore.reset()

    def add_documents(self, documents: list[str]) -> None:
        """Index documents using document-aware pgvector writes."""
        for document in documents:
            path = Path(document)
            items = self.indexer.index_file(path)
            self.datastore.add_document(path, self.indexer.file_hash(path), items)
            print(f"Added {len(items)} chunks from {path.name}")

    def answer_query(
        self, query: str, history: list[dict[str, str]] | None = None
    ) -> tuple[str, list[SearchResult]]:
        search_results = self.retriever.search(query)
        response = self.response_generator.generate_response(query, search_results, history)
        return response, search_results

    def process_query(self, query: str) -> str:
        response, _ = self.answer_query(query)
        return response

    def evaluate(self, sample_questions: list[dict[str, str]]) -> list[EvaluationResult]:
        # Evaluate a list of questions/answer pairs
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results: list[EvaluationResult] = list(
                executor.map(
                    self._evaluate_single_question,
                    questions,
                    expected_answers,
                )
            )

        for i, result in enumerate(results):
            result_emoji = "✅" if result.is_correct else "❌"
            print(f"{result_emoji} Q {i + 1}: {result.question}: \n")
            print(f"Response: {result.response}\n")
            print(f"Expected Answer: {result.expected_answer}\n")
            print(f"Reasoning: {result.reasoning}\n")
            print("--------------------------------")

        number_correct = sum(result.is_correct for result in results)
        print(f" Total Score: {number_correct}/{len(results)}")
        return results

    def _evaluate_single_question(self, question: str, expected_answer: str) -> EvaluationResult:
        # Evaluate a single question/answer pair
        response = self.process_query(question)
        return self.evaluator.evaluate(question, response, expected_answer)
