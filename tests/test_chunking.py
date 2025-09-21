from src.ingestion.chunking import chunk_words

def test_chunking_basic():
    text = "one two three four five six seven eight nine ten"
    chunks = list(chunk_words(text, chunk_size=4, overlap=1))
    assert len(chunks) >= 2