from pathlib import Path

from src.impl.indexer import Indexer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_all_data_files_are_discovered_and_chunked() -> None:
    indexer = Indexer()
    paths = indexer.discover(DATA_DIR)

    assert [path.name for path in paths] == [
        "ec2-api.md",
        "vpc-pg.md",
        "vpc-tm.md",
        "vpc-ug.md",
    ]
    for path in paths:
        items = indexer.index_file(path)
        assert items, f"No chunks created for {path.name}"
        assert all(item.content.strip() for item in items)
        assert all(len(item.content) <= indexer.max_chars for item in items)
        assert len({item.source for item in items}) == len(items)
        assert [item.chunk_index for item in items] == list(range(len(items)))


def test_markdown_headings_are_carried_into_chunk_metadata() -> None:
    items = Indexer().index_file(DATA_DIR / "vpc-pg.md")
    assert any("VPC Peering" in item.heading for item in items)


def test_plain_text_fallback_chunks_large_api_reference() -> None:
    items = Indexer().index_file(DATA_DIR / "ec2-api.md")
    assert len(items) > 100
    assert all(item.heading == "" for item in items)


def test_common_pdf_mojibake_is_repaired() -> None:
    indexer = Indexer()
    assert indexer._normalize("Copyright Â© 2026") == "Copyright © 2026"
