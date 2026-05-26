"""File loaders. Only the formats we actually need for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    id: str          # stable id derived from the path
    source: str      # human-readable origin
    text: str
    metadata: dict


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def load_path(path: Path) -> list[LoadedDocument]:
    """Load a single file or every supported file under a directory."""
    if path.is_file():
        return [_load_file(path)]
    if path.is_dir():
        docs: list[LoadedDocument] = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.append(_load_file(f))
        return docs
    raise FileNotFoundError(f"No such file or directory: {path}")


def _load_file(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    else:
        text = path.read_text(encoding="utf-8")
    return LoadedDocument(
        id=str(path.as_posix()),
        source=path.name,
        text=text,
        metadata={"path": str(path), "suffix": suffix},
    )


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)
