from pathlib import Path
from pypdf import PdfReader
import docx2txt
from src.utils.io import write_jsonl

def extract_pdf(p: Path) -> str:
    reader = PdfReader(p)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_docx(p: Path) -> str:
    return docx2txt.process(p)

def extract_any(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        return extract_pdf(p)
    if p.suffix.lower() == ".docx":
        return extract_docx(p)
    if p.suffix.lower() in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore")
    return ""

if __name__ == "__main__":
    print("Use pipeline.py for end-to-end ingestion.")