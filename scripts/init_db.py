import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenAIEmbeddings
from langchain_community.vectorstores import Qdrant

def initialize_and_seed_db():
    print("--- STARTING LOCAL DATABASE SEEDING ---")
    
    # Fetch API Key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "YourActual" in api_key:
        raise ValueError("Please export a valid GEMINI_API_KEY in your terminal session first!")

    # Using Google's native text embedding model
    embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

    # Simulated enterprise knowledge base
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

    # Initialize Qdrant completely locally inside a simple storage directory
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_qdrant_storage")
    
    print(f"Creating offline persistent database vector storage inside: {db_path}")
    
    vector_store = Qdrant.from_documents(
        documents=seed_data,
        embedding=embeddings,
        path=db_path,
        collection_name="ey_internal_docs"
    )
    
    print("--- DATABASE SEEDED SUCCESSFULLY WITH 3 ENTERPRISE RECORDS ---")

if __name__ == "__main__":
    initialize_and_seed_db()