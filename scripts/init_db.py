import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def load_local_settings_keys():
    """Automatically extracts the Gemini API Key, embedding model, and
    Qdrant collection name from local.settings.json"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    settings_path = os.path.join(base_dir, "local.settings.json")

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Could not locate local.settings.json at {settings_path}")

    with open(settings_path, "r") as f:
        data = json.load(f)

    values = data.get("Values", {})
    api_key = values.get("GEMINI_API_KEY")
    model_name = values.get("GEMINI_EMB_MODEL")
    collection_name = values.get("QDRANT_COLLECTION_NAME")

    if not api_key or "YourActual" in api_key:
        raise ValueError("GEMINI_API_KEY inside local.settings.json is empty or contains placeholder text!")

    if not model_name:
        raise ValueError("GEMINI_EMB_MODEL is missing from local.settings.json")

    if not collection_name:
        raise ValueError(
            "QDRANT_COLLECTION_NAME is missing from local.settings.json. "
            "Add it under Values, e.g. \"QDRANT_COLLECTION_NAME\": \"knowledge_base\""
        )

    print("--- SUCCESS: Loaded GEMINI_API_KEY, GEMINI_EMB_MODEL, and QDRANT_COLLECTION_NAME ---")
    return api_key, model_name, collection_name


def initialize_db():
    print("--- STARTING LOCAL DATABASE INITIALIZATION ---")

    gemini_key, model_name, collection_name = load_local_settings_keys()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=gemini_key
    )

    # Qdrant needs to know the vector dimension up front to create a
    # collection. Rather than hardcoding a number that could go stale if
    # the embedding model is ever swapped, we embed one throwaway string
    # purely to learn the dimension. Nothing from this call is stored.
    print("Determining embedding dimension from the configured model...")
    probe_vector = embeddings.embed_query("dimension probe - not stored")
    vector_size = len(probe_vector)
    print(f"Embedding model '{model_name}' produces {vector_size}-dimensional vectors.")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, "local_qdrant_storage")
    print(f"Initializing offline persistent vector index files inside: {db_path}")

    client = QdrantClient(path=db_path)

    if client.collection_exists(collection_name):
        print(f"--- Collection '{collection_name}' already exists. Nothing to do. ---")
        print("If you want a completely fresh start, delete the local_qdrant_storage folder first.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    print(f"--- DATABASE INITIALIZED: empty collection '{collection_name}' created ---")
    print("No documents were added. Use the POST /api/documents endpoint (or the")
    print("blob ingestion script) to populate it.")


if __name__ == "__main__":
    initialize_db()