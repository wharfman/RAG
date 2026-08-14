from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from ftfy import fix_text

from src.interface.base_datastore import DataItem
from src.interface.base_indexer import BaseIndexer

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


class Indexer(BaseIndexer):
    """Chunk Markdown and text files without losing section boundaries."""

    def __init__(self, max_chars: int = 3600, overlap_chars: int = 320):
        if max_chars < 500 or overlap_chars >= max_chars:
            raise ValueError("Invalid chunk size configuration")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def index(self, document_paths: list[str]) -> list[DataItem]:
        items: list[DataItem] = []
        for document_path in document_paths:
            items.extend(self.index_file(Path(document_path)))

        print(f"Created {len(items)} items from {len(document_paths)} documents.")
        return items

    def discover(self, data_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def index_file(self, path: Path) -> list[DataItem]:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {path.suffix}")
        raw = path.read_text(encoding="utf-8-sig")
        text = self._normalize(raw)
        if not text.strip():
            raise ValueError(f"Document is empty: {path}")

        chunks = self._chunk_text(text)
        items: list[DataItem] = []
        for chunk_index, (heading, content) in enumerate(chunks):
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            items.append(
                DataItem(
                    content=content,
                    source=f"{path.name}#chunk-{chunk_index + 1}",
                    file_name=path.name,
                    heading=heading,
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                )
            )
        return items

    @staticmethod
    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _normalize(text: str) -> str:
        text = fix_text(text).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _chunk_text(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        headings = [""] * 6
        current_heading = ""
        current_lines: list[str] = []

        def flush_section() -> None:
            nonlocal current_lines
            if any(line.strip() for line in current_lines):
                sections.append((current_heading, current_lines))
            current_lines = []

        for line in text.split("\n"):
            match = HEADING_RE.match(line)
            if match and match.group(2).strip("* _"):
                flush_section()
                level = len(match.group(1))
                title = match.group(2).strip().strip("*").strip()
                headings[level - 1] = title
                headings[level:] = [""] * (6 - level)
                current_heading = " > ".join(value for value in headings if value)
                current_lines.append(line)
            else:
                current_lines.append(line)
        flush_section()

        chunks: list[tuple[str, str]] = []
        buffer = ""
        buffer_heading = ""
        for heading, lines in sections:
            units = self._units(lines)
            for unit in units:
                for piece in self._split_oversized(unit):
                    if not buffer:
                        buffer_heading = heading
                    elif not buffer_heading and heading:
                        buffer_heading = heading
                    candidate = piece if not buffer else f"{buffer}\n{piece}"
                    if len(candidate) <= self.max_chars:
                        buffer = candidate
                        continue
                    if buffer.strip():
                        chunks.append((buffer_heading, buffer.strip()))
                    overlap = self._tail(buffer)
                    available_overlap = self.max_chars - len(piece) - 1
                    if available_overlap <= 0:
                        overlap = ""
                    elif len(overlap) > available_overlap:
                        overlap = overlap[-available_overlap:].lstrip()
                    buffer = f"{overlap}\n{piece}".strip() if overlap else piece
                    buffer_heading = heading
        if buffer.strip():
            chunks.append((buffer_heading, buffer.strip()))
        return chunks

    @staticmethod
    def _units(lines: Iterable[str]) -> list[str]:
        units: list[str] = []
        paragraph: list[str] = []
        for line in lines:
            if line.strip():
                paragraph.append(line)
            elif paragraph:
                units.append("\n".join(paragraph))
                paragraph = []
        if paragraph:
            units.append("\n".join(paragraph))
        return units

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        pieces: list[str] = []
        remaining = text
        while len(remaining) > self.max_chars:
            cut = remaining.rfind("\n", 0, self.max_chars + 1)
            if cut < self.max_chars // 2:
                cut = remaining.rfind(" ", 0, self.max_chars + 1)
            if cut < self.max_chars // 2:
                cut = self.max_chars
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    def _tail(self, text: str) -> str:
        if not text or self.overlap_chars == 0:
            return ""
        tail = text[-self.overlap_chars :]
        boundary = max(tail.find("\n"), tail.find(" "))
        return tail[boundary + 1 :].strip() if boundary >= 0 else tail.strip()
