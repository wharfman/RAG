from abc import ABC, abstractmethod

from .base_datastore import SearchResult


class BaseResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: list[SearchResult]) -> str:
        pass
