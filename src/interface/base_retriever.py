from abc import ABC, abstractmethod

from .base_datastore import SearchResult


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        pass
