from pathlib import Path
import json

def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_first_lines(path: Path, n=3):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            out.append(line.strip())
    return out