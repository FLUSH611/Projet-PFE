# src/rag/retriever.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./vectors")
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_retriever(k: int = 4):
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
