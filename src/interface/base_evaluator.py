from abc import ABC, abstractmethod

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    question: str
    response: str
    expected_answer: str
    is_correct: bool
    reasoning: str | None = None  # Debug AI response


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult:
        pass
