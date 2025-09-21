import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"
VEC_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "vectors"))
OUT_DIR = ROOT / "out"
LOG_DIR = ROOT / "logs"

COLLECTION_NAME = "itstorm_docs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")