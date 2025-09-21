from sentence_transformers import SentenceTransformer
from src.config.settings import EMBEDDING_MODEL

_MODEL = None

def get_embedder():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL