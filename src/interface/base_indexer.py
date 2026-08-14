from abc import ABC, abstractmethod

from .base_datastore import DataItem


class BaseIndexer(ABC):
    @abstractmethod
    def index(self, document_paths: list[str]) -> list[DataItem]:
        pass
