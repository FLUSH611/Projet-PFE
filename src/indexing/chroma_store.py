import chromadb
from chromadb.config import Settings
from src.config.settings import VEC_DIR, COLLECTION_NAME

def get_client():
    return chromadb.Client(Settings(persist_directory=str(VEC_DIR), is_persistent=True))

def reset_collection(client=None, name=COLLECTION_NAME):
    client = client or get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)

def get_collection(client=None, name=COLLECTION_NAME):
    client = client or get_client()
    return client.get_collection(name)