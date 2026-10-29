import os
from typing import List, TypedDict, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from core.ticketing import create_ticket


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    route_choice: str
    ticket_id: Optional[str]


# Structured validation schema for the routing decision.
# Pydantic forces the model's routing output into a strict literal
# instead of free-form text we'd have to parse ourselves.
class RouterSchema(BaseModel):
    datasource: Literal["vectorstore", "fallback"] = Field(
        description="Route to 'vectorstore' if the query could plausibly be answered using the indexed knowledge base. Otherwise route to 'fallback'."
    )


class KnowledgeBaseAgent:
    """
    Domain-agnostic RAG agent: determine_route -> either
    (retrieve_docs -> generate_answer) or execute_fallback.

    Works against whatever is stored in the configured Qdrant collection -
    no assumptions are made here about what kind of documents it holds
    (IT policies, research papers, or anything else).

    Whenever neither path produces a usable answer (query routed as
    unrelated, or routed correctly but nothing relevant was retrieved),
    a support ticket is automatically created via core.ticketing so the
    question doesn't just disappear into a dead-end response.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_EMB_MODEL")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        base_dir = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, "local_qdrant_storage")

        if not collection_name:
            raise ValueError(
                "QDRANT_COLLECTION_NAME is missing from local.settings.json. "
                "It must match the name used when the collection was created via scripts/init_db.py."
            )

        # NOTE: this MUST match the model used in scripts/init_db.py exactly,
        # or retrieval will silently break (dimension mismatch / garbage results).
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )

        self.client = QdrantClient(path=db_path)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name=collection_name
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
        self.structured_router = self.llm.with_structured_output(RouterSchema)

    def determine_route(self, question: str) -> str:
        system_prompt = (
            f"Analyze this user query: '{question}' and decide whether it could plausibly "
            f"be answered using documents in a knowledge base, or whether it's unrelated/general "
            f"and should fall back to a default response."
        )
        try:
            decision = self.structured_router.invoke(system_prompt)
            return decision.datasource
        except Exception:
            # Fallback gracefully if model/serialization stumbles
            return "fallback"

    def retrieve_docs(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def generate_answer(self, question: str, docs: List[Document]) -> Tuple[str, Optional[str]]:
        """Returns (answer_text, ticket_id_or_none)."""
        if not docs:
            ticket = create_ticket(question, reason="no_relevant_documents")
            answer = (
                f"FALLBACK_TRIGGERED: No relevant documents met the retrieval threshold. "
                f"A ticket has been created for follow-up: {ticket['ticket_id']}"
            )
            return answer, ticket["ticket_id"]

        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            f"You are a knowledge-base assistant. Answer the user's question using ONLY "
            f"the provided context below. If the context does not contain enough information "
            f"to answer confidently, say so explicitly rather than guessing.\n\n"
            f"User Question: {question}\n\n"
            f"Retrieved Context:\n{context}"
        )
        response = self.llm.invoke(prompt)
        return response.content, None

    def execute_fallback(self, question: str) -> Tuple[str, str]:
        """Returns (answer_text, ticket_id)."""
        ticket = create_ticket(question, reason="routed_as_unrelated")
        answer = (
            f"FALLBACK_TRIGGERED: No matching information was found in the knowledge base "
            f"for this query. A ticket has been created for follow-up: {ticket['ticket_id']}"
        )
        return answer, ticket["ticket_id"]

    def run(self, question: str) -> AgentState:
        route_choice = self.determine_route(question)

        if route_choice == "vectorstore":
            docs = self.retrieve_docs(question)
            generation, ticket_id = self.generate_answer(question, docs)
        else:
            docs = []
            generation, ticket_id = self.execute_fallback(question)

        return {
            "question": question,
            "documents": docs,
            "generation": generation,
            "route_choice": route_choice,
            "ticket_id": ticket_id,
        }


# --- Cached agent instance ---
# Qdrant's embedded (path=) mode takes an exclusive file lock on db_path.
# Building a new QdrantClient on every HTTP request risks "storage folder
# already accessed by another instance" errors once concurrent/sequential
# requests overlap. Build the agent once and reuse it.
_agent_cache = None


def get_agent() -> KnowledgeBaseAgent:
    global _agent_cache
    if _agent_cache is None:
        _agent_cache = KnowledgeBaseAgent()
    return _agent_cache