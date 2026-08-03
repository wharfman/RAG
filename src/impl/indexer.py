import os
from typing import List
from interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConvertor
from docling.chunking import HybridChunker, DocChunk


class Indexer(BaseIndexer):

    def index(self, document_paths) -> List[DataIem]:
        items = []
        for document_path in document_paths:
            new_items = self._simple_pdf_chunker(document_path)
            items.extend(new_items)

        print(
            f"Created {len(items)} itmes from {len(document_paths)} documents."
        )
        return items

    def _simple_pdf_chunker(self, pdf_path, chunk_size) -> List[DataItem]:
        items = []
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PDFReader(pdf_file)
            doc_index = 0

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.exctract_text().strip().replace("\n", "")
                start_index = 0

                while start_index < len(page_text):
                    end_index = min(start_index + chunk_size, len(page_text))
                    chunk = page_text[start_index:end_index]
                    source = f"{pdf_path}:{doc_index}"
                    item =  DataItem(content=chunk, source=source)
                    items.append(item)
                    start_index = end_index
                    doc_index += 1

        return items