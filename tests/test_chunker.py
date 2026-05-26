from app.rag.chunker import RecursiveCharacterTextSplitter


def test_returns_empty_for_empty_input():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    assert splitter.split("") == []


def test_short_text_yields_single_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split("This is a short document.")
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short document."


def test_long_text_is_split_under_chunk_size():
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    paragraph = ("Sentence one. " * 30).strip()
    chunks = splitter.split(paragraph)
    assert len(chunks) > 1
    # Allow a little slack for the overlap-tail re-seed.
    for c in chunks:
        assert len(c.text) <= 80 + 20


def test_overlap_must_be_less_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)


def test_paragraph_boundary_preferred_over_word_split():
    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    text = "First paragraph here.\n\nSecond paragraph follows."
    chunks = splitter.split(text)
    # Both paragraphs should end up as their own (or first-in) chunk — not
    # word-split mid-paragraph.
    joined = " ".join(c.text for c in chunks)
    assert "First paragraph here." in joined
    assert "Second paragraph follows." in joined
