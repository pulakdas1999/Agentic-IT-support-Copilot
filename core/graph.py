import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    route_choice: str
    
class RouterSchema(BaseModel):
    datasource: Literal["vectorstore", "fallback"] = Field(
        description="Route to 'vectorstore' if query relates to corporate operations, IT, VPN, or hardware. Otherwise route to 'fallback'."
    )

class ITSupportAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_EMB_MODEL")
        base_dir = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, "local_qdrant_storage")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )

        self.client = QdrantClient(path=db_path)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name="ey_internal_docs"
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
        self.structured_router = self.llm.with_structured_output(RouterSchema)

    def determine_route(self, question: str) -> str:
        system_prompt = f"Analyze this enterprise user query: '{question}' and decide its destination data stack."
        try:
            decision = self.structured_router.invoke(system_prompt)
            return decision.datasource
        except Exception:
            return "fallback"

    def retrieve_docs(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        if not docs:
            return "FALLBACK_TRIGGERED: No relevant local documents met the certainty filter."

        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            f"You are an EY support assistant. Answer the user query using ONLY the provided verified context.\n"
            f"User Question: {question}\n\n"
            f"Verified Enterprise Context:\n{context}"
        )
        response = self.llm.invoke(prompt)
        return response.content

    def execute_fallback(self) -> str:
        return "FALLBACK_TRIGGERED: Query details not found in local internal corporate records. Ticket routed to the live IT Support queue."

    def run(self, question: str) -> AgentState:
        route_choice = self.determine_route(question)

        if route_choice == "vectorstore":
            docs = self.retrieve_docs(question)
            generation = self.generate_answer(question, docs)
        else:
            docs = []
            generation = self.execute_fallback()

        return {
            "question": question,
            "documents": docs,
            "generation": generation,
            "route_choice": route_choice,
        }

_agent_cache = None
def get_agent() -> ITSupportAgent:
    global _agent_cache
    if _agent_cache is None:
        _agent_cache = ITSupportAgent()
    return _agent_cache