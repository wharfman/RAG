import lancedb
from openai import OpenAI
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor


class Datastore:

    DB_PATH = "data/sample-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        self.vector_dimensions = 1536 # Match OpenAI embedding model 
        self.open_ai_client = OpenAI()
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table = self._get_table() # Open or creates table

    def reset(self) -> Table:
        try:
            self.vector_db.drop_table(self.DB_TABLE_NAME)
        except Exception as e:
            print("Unable to drop table.")


        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self.vector_dimensions)),
                pa.field("content", pa.utf8()),
                pa.field("source", pa.utf8()),
            ]
        )
        self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        self.table = self.vector_db.open_table(self.DB_TABLE_NAME)
        print(f" Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.table

    def get_vector(self, content: str) -> List[float]:
        response = self.open_ai_client.embeddings.create(
            input = content,
            model = "text-embedding-3-small",
            dimensions = self.vector_dimensions,
        )
        embeddings = response.data[0].embedding
        return embeddings

    def add_items(self, items: List[DataItem]) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            entries = list(executor.map(self._convert_item_to_entry, items))

        self.table.merge_insert(
            "source"
        ).when_matched_update_all().when_not_matched_insert_all().execute(entries)

    def search(self, query, top_k) -> List[str]:
        vector = self.get_vector(query)
        results = (
            self.table.search(vector)
            .select(["content", "source"])
            .limit(top_k)
            .to_list()
        )

        result_content = [result.get("content") for result in results]
        return result_content

    def _get_table(self) -> Table:
        try:
            return self.vector_db.open_table(self.DB_TABLE_NAME)
        except Exception as e:
            print(f"Error opening table. Try resetting the datastore: {e}")
            return self.reset()

    def _conver_item_to_entry(self, item) -> dict:
        """Convert a Dataitem to match table schema."""
        vector = self.get_vector(item.content)
        return {
            "vector": vector,
            "content": item.content,
            "source": item.source,
        }