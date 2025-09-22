# src/rag/llm.py
import os
from dotenv import load_dotenv

# Load your custom env file
load_dotenv(dotenv_path="r.env", override=True)


def get_mistral_llm_or_none():
    """Return a ChatMistralAI instance if key + model are valid, else None."""
    try:
        from langchain_mistralai import ChatMistralAI
    except Exception:
        return None

    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        return None

    model = os.getenv("LLM_MODEL", "mistral-small-latest")

    # Fallback protection
    if not model.startswith("mistral"):
        print(f"[WARN] Invalid model '{model}', switching to mistral-small-latest")
        model = "mistral-small-latest"

    try:
        return ChatMistralAI(
            model=model,
            temperature=0.2,
            max_tokens=384,
            api_key=api_key,
        )
    except Exception:
        return None


def get_llm():
    """Always return a ChatMistralAI instance or raise an error."""
    from langchain_mistralai import ChatMistralAI

    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing!")

    model = os.getenv("LLM_MODEL", "mistral-small-latest")

    if not model.startswith("mistral"):
        print(f"[WARN] Invalid model '{model}', switching to mistral-small-latest")
        model = "mistral-small-latest"

    return ChatMistralAI(
        model=model,
        temperature=0.2,
        max_tokens=384,
        api_key=api_key,
    )
