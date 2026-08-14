from src.interface.base_datastore import BaseDatastore, SearchResult
from src.interface.base_retriever import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 6) -> list[SearchResult]:
        return self.datastore.search(query, top_k=top_k)
