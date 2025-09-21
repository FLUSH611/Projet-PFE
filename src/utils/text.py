import re
def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()