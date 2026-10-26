import os
import json
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

def load_local_settings_keys():
    """Automatically extracts the Gemini API Key from local.settings.json"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    settings_path = os.path.join(base_dir, "local.settings.json")
    
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Could not locate local.settings.json at {settings_path}")
        
    with open(settings_path, "r") as f:
        data = json.load(f)
        
    values = data.get("Values", {})
    api_key = values.get("GEMINI_API_KEY")
    
    if not api_key or "YourActual" in api_key:
        raise ValueError("GEMINI_API_KEY inside local.settings.json is empty or contains placeholder text!")
        
    print("--- SUCCESS: Loaded GEMINI_API_KEY from local.settings.json ---")
    return api_key

def initialize_and_seed_db():
    print("--- STARTING LOCAL DATABASE SEEDING ---")
    
    # 1. Load the api key string explicitly from file
    gemini_key = load_local_settings_keys()

    # Ensure it includes the "models/" prefix and matching model identifier number
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_key
    )


    # 3. Simulated enterprise knowledge base records
    seed_data = [
        Document(
            page_content="EY internal VPN configuration requires the GlobalProtect Client v5.2. Multi-factor authentication (MFA) must be completed using the Microsoft Authenticator app connected to your corporate ID.", 
            metadata={"category": "IT_Infrastructure", "severity": "High"}
        ),
        Document(
            page_content="To request a new corporate laptop hardware upgrade, employees must submit an asset allocation ticket via the Talent Marketplace portal after completing 24 months of continuous service.", 
            metadata={"category": "HR_Hardware", "severity": "Low"}
        ),
        Document(
            page_content="Password policy rules: EY domain passwords expire every 90 days. New passwords must contain at least 14 characters, including special symbols, and cannot match the last 5 passwords used.", 
            metadata={"category": "Security_Policy", "severity": "Critical"}
        )
    ]

    # 4. Initialize Qdrant completely locally inside your clean workspace folder tree
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_qdrant_storage")
    print(f"Compiling offline persistent database vector index files inside: {db_path}")
    
    vector_store = QdrantVectorStore.from_documents(
        documents=seed_data,
        embedding=embeddings,
        path=db_path,
        collection_name="ey_internal_docs"
    )
    
    print("--- DATABASE SEEDED SUCCESSFULLY WITH 3 ENTERPRISE RECORDS ---")

if __name__ == "__main__":
    initialize_and_seed_db()