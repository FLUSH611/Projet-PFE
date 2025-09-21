from pathlib import Path
from src.config.settings import RAW_DIR, PROC_DIR
from src.ingestion.extract_text import extract_any
from src.ingestion.chunking import chunk_words
from src.utils.io import write_jsonl

def run(in_dir=RAW_DIR, out_path=PROC_DIR / "chunks.jsonl"):
    records = []
    for p in Path(in_dir).glob("*"):
        text = extract_any(p)
        if not text: continue
        for ch in chunk_words(text):
            records.append({"source": str(p), "text": ch})
    write_jsonl(records, out_path)
    print(f"Saved {len(records)} chunks -> {out_path}")

if __name__ == "__main__":
    run()