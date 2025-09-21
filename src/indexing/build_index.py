import json
from src.indexing.embeddings import get_embedder
from src.indexing.chroma_store import reset_collection
from src.config.settings import PROC_DIR

def run(in_path=PROC_DIR / "chunks.jsonl"):
    coll = reset_collection()
    model = get_embedder()

    texts, metas, ids = [], [], []
    with open(in_path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            texts.append(r["text"])
            metas.append({"source": r["source"]})
            ids.append(f"id_{k}")
    emb = model.encode(texts, normalize_embeddings=True).tolist()
    coll.add(documents=texts, embeddings=emb, metadatas=metas, ids=ids)
    print(f"Indexed {len(texts)} chunks.")

if __name__ == "__main__":
    run()