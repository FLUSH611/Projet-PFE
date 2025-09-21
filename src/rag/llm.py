# src/rag/llm.py
import os
from dotenv import load_dotenv
load_dotenv()

def get_mistral_llm_or_none():
    try:
        from langchain_mistralai import ChatMistralAI
    except Exception:
        return None

    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        return None

    model = os.getenv("LLM_MODEL", "mistral-small")
    try:
        return ChatMistralAI(model=model, temperature=0.2, max_tokens=384, api_key=api_key)
    except Exception:
        return None

def get_llm():
    """
    Retourne un LLM utilisable par la chaîne :
    - Mistral API si clé dispo
    - Sinon tente un petit modèle local (optionnel)
    - Sinon retourne None (la chaîne fera un fallback sur les extraits)
    """
    llm = get_mistral_llm_or_none()
    if llm is not None:
        return llm
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline
        model_name = os.getenv("LOCAL_LLM", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline("text-generation", model=mdl, tokenizer=tok,
                       max_new_tokens=384, do_sample=False, temperature=0.2,
                       repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        return HuggingFacePipeline(pipeline=gen)
    except Exception:
        return None
