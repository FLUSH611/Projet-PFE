# src/rag/chain.py
from typing import Dict, Any
from pathlib import Path
from src.rag.retriever import get_retriever
from src.rag.llm import get_llm

SYSTEM_PROMPT = (
    "Tu es un assistant de consulting. Réponds de façon concise, en citant les sources (nom de fichier + chunk). "
    "Si l'information n'est pas présente dans les sources, dis-le explicitement."
)

def _format_docs(docs):
    parts, sources = [], []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", "?")
        sources.append(f"{Path(src).name}#chunk{cid}")
        parts.append(f"[Source: {Path(src).name}] {d.page_content}")
    return "\n\n".join(parts), list(dict.fromkeys(sources))

def build_rag_chain(k: int = 4):
    retriever = get_retriever(k=k)
    llm = get_llm()

    class RagCallable:
        def invoke(self, question: str) -> Dict[str, Any]:
            docs = retriever.get_relevant_documents(question)
            context_text, sources = _format_docs(docs)
            if llm is None:
                # Fallback Mois 1 : pas de génération → renvoyer les extraits pertinents
                return {
                    "answer": "⚠️ Aucun LLM configuré. Voici les extraits pertinents :\n\n" + context_text[:2000],
                    "sources": sources,
                }
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {question}\n"
                f"Contextes:\n{context_text}\n\n"
                f"Réponse:"
            )
            try:
                resp = llm.invoke(prompt)
                text = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                text = f"Erreur LLM: {e}\n\nExtraits:\n{context_text[:1500]}"
            return {"answer": text, "sources": sources}

    return RagCallable()
