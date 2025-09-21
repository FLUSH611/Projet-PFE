# src/rag/quick_retrieve.py
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()
DB_DIR = os.getenv("CHROMA_DIR", "./vectors")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.Client(Settings(persist_directory=DB_DIR, is_persistent=True))
try:
    # Par défaut Chroma CLI/SDK crée "langchain" si tu as indexé via LangChain
    coll = client.get_or_create_collection("langchain")
except Exception as e:
    raise SystemExit(f"Chroma error: {e}. As-tu construit l'index ?")

embedder = SentenceTransformer(EMB_MODEL)
query = input("Type a test question: ")
q = embedder.encode([query], normalize_embeddings=True).tolist()[0]
res = coll.query(query_embeddings=[q], n_results=3)

for i, (doc, meta) in enumerate(zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]), 1):
    print(f"\n#{i} Source: {meta.get('source')}")
    print(doc[:300] + ("..." if len(doc) > 300 else ""))
