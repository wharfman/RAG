from abc import ABC, abstractmethod

from pydantic import BaseModel


class DataItem(BaseModel):
    content: str = ""
    source: str = ""
    file_name: str = ""
    heading: str = ""
    chunk_index: int = 0
    content_hash: str = ""


class SearchResult(BaseModel):
    content: str
    source: str
    file_name: str
    heading: str = ""
    score: float


class BaseDatastore(ABC):
    @abstractmethod
    def add_items(self, items: list[DataItem]) -> None:
        pass

    @abstractmethod
    def get_vector(self, content: str) -> list[float]:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        pass
