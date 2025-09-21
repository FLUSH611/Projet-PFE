def chunk_words(text: str, chunk_size=900, overlap=150):
    words = text.split()
    i = 0
    while i < len(words):
        yield " ".join(words[i:i+chunk_size])
        i += (chunk_size - overlap)