import json, os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_DIR = "vectors"
COLL = "itstorm_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main(in_path="data/processed/chunks.jsonl"):
    os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.Client(Settings(persist_directory=DB_DIR, is_persistent=True))

    # Supprimer la collection si elle existe
    try:
        client.delete_collection(COLL)
    except Exception:
        pass

    coll = client.create_collection(COLL)
    model = SentenceTransformer(MODEL_NAME)

    texts, metadatas, ids = [], [], []
    with open(in_path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            rec = json.loads(line)
            texts.append(rec["text"])
            metadatas.append({"source": rec["source"]})
            ids.append(f"id_{k}")

    emb = model.encode(texts, normalize_embeddings=True).tolist()
    coll.add(documents=texts, embeddings=emb, metadatas=metadatas, ids=ids)
    print(f"✅ Indexed {len(texts)} chunks in Chroma collection '{COLL}'")

if __name__ == "__main__":
    main()
