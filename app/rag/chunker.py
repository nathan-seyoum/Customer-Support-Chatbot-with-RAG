"""Recursive character text splitter.

Implements the same algorithm popularized by LangChain's
`RecursiveCharacterTextSplitter`: split on the most semantic boundary that
keeps chunks under `chunk_size`, falling back to less semantic boundaries
until a fixed-width split is the last resort. Adjacent chunks share
`chunk_overlap` characters so an entity that straddles a boundary is still
recoverable.

We implement it ourselves (about 50 lines) rather than pulling in the whole
LangChain dependency tree. The algorithm is well-known and easier to reason
about / test in-tree.
"""

from __future__ import annotations

from dataclasses import dataclass


# Order matters: most semantic boundary first.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # sentence
    "? ",
    "! ",
    "; ",
    ", ",
    " ",      # word
    "",       # character (last resort)
)


@dataclass
class TextChunk:
    text: str
    start: int  # character offset in the original document
    end: int


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    # ------------------------------------------------------------------ public
    def split(self, text: str) -> list[TextChunk]:
        if not text:
            return []
        pieces = self._split_recursive(text, list(self.separators))
        return self._merge_with_overlap(pieces, text)

    # ----------------------------------------------------------------- helpers
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Split `text` on the first separator that produces sub-pieces; recurse
        into any piece still larger than `chunk_size`."""
        if len(text) <= self.chunk_size:
            return [text]

        for i, sep in enumerate(separators):
            if sep == "":
                # Last-resort fixed-width split.
                return [text[j : j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:
                rest = separators[i + 1 :]
                out: list[str] = []
                for piece in text.split(sep):
                    if not piece:
                        continue
                    if len(piece) <= self.chunk_size:
                        out.append(piece)
                    else:
                        out.extend(self._split_recursive(piece, rest))
                return out
        return [text]

    def _merge_with_overlap(self, pieces: list[str], original: str) -> list[TextChunk]:
        """Greedy merge pieces back together up to `chunk_size`, then slide the
        window by `chunk_size - chunk_overlap`. Character offsets are
        approximate (best-effort) so downstream code can locate the chunk."""
        if not pieces:
            return []

        chunks: list[TextChunk] = []
        buf: list[str] = []
        buf_len = 0
        cursor = 0  # running offset in original text

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            sep = " " if buf else ""
            if buf_len + len(sep) + len(piece) <= self.chunk_size:
                buf.append(piece)
                buf_len += len(sep) + len(piece)
            else:
                chunk_text = " ".join(buf).strip()
                if chunk_text:
                    start = original.find(chunk_text[: min(40, len(chunk_text))], cursor)
                    if start < 0:
                        start = cursor
                    chunks.append(TextChunk(text=chunk_text, start=start, end=start + len(chunk_text)))
                    cursor = max(cursor, start + len(chunk_text) - self.chunk_overlap)
                # Re-seed buffer with overlap tail of the just-emitted chunk.
                if chunks and self.chunk_overlap > 0:
                    tail = chunks[-1].text[-self.chunk_overlap :]
                    buf = [tail, piece]
                    buf_len = len(tail) + 1 + len(piece)
                else:
                    buf = [piece]
                    buf_len = len(piece)

        if buf:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                start = original.find(chunk_text[: min(40, len(chunk_text))], cursor)
                if start < 0:
                    start = cursor
                chunks.append(TextChunk(text=chunk_text, start=start, end=start + len(chunk_text)))

        return chunks
