from interface.base_retriever import BaseRetriever
from interface.base_datastore import BaseDataStore

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDataStore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 3) -> list[str]:
        # Delegate search to underlying datastore
        search_results = self.datastore.search(query, top_k=top_k * 3)
        return search_results