"""
Standalone viewer for the local Qdrant vector DB.

IMPORTANT: Qdrant's embedded (path=) mode takes an exclusive file lock.
Run this ONLY when `func start` is NOT running, otherwise you'll get a
"storage folder already accessed by another instance" error.

Usage:
    python scripts/view_db.py
"""
import os
import json
from qdrant_client import QdrantClient


def load_collection_name() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    settings_path = os.path.join(base_dir, "local.settings.json")

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Could not locate local.settings.json at {settings_path}")

    with open(settings_path, "r") as f:
        data = json.load(f)

    collection_name = data.get("Values", {}).get("QDRANT_COLLECTION_NAME")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION_NAME is missing from local.settings.json")

    return collection_name


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, "local_qdrant_storage")
    collection_name = load_collection_name()

    if not os.path.exists(db_path):
        print(f"No database found at {db_path}. Run scripts/init_db.py first.")
        return

    client = QdrantClient(path=db_path)

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"Collection '{collection_name}' not found. Available: {collections}")
        return

    count = client.count(collection_name=collection_name).count
    print(f"--- Collection '{collection_name}' contains {count} document(s) ---\n")

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=count or 100,
        with_payload=True,
        with_vectors=False,  # set True if you want to inspect raw embedding floats
    )

    output = []
    for p in points:
        output.append({
            "id": p.id,
            "page_content": p.payload.get("page_content"),
            "metadata": p.payload.get("metadata"),
        })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()