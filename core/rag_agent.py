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
    sources: List[dict]


# Structured validation schema for the routing decision.
# Pydantic forces the model's routing output into a strict literal
# instead of free-form text we'd have to parse ourselves.
class RouterSchema(BaseModel):
    datasource: Literal["vectorstore", "fallback"] = Field(
        description="Route to 'vectorstore' if the query could plausibly be answered using the indexed knowledge base. Otherwise route to 'fallback'."
    )


# How much of each chunk's text to include in the "sources" excerpt
# returned to the caller. This is just a preview for display purposes -
# the full chunk text is still what's sent to the LLM for generation.
SOURCE_EXCERPT_LENGTH = 200


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

    When an answer IS generated, every retrieved chunk's source metadata
    is returned alongside it (see `sources` in the result), so the caller
    can show exactly which document(s) and chunk(s) backed the answer -
    the answer is never just taken on faith.
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

    def _build_sources(self, docs: List[Document]) -> List[dict]:
        """
        One entry per retrieved chunk, in the same order they were
        numbered in the prompt sent to the LLM (source_number matches the
        [N] markers the model was asked to cite with).
        """
        sources = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            excerpt = doc.page_content[:SOURCE_EXCERPT_LENGTH]
            if len(doc.page_content) > SOURCE_EXCERPT_LENGTH:
                excerpt += "..."

            sources.append({
                "source_number": idx,
                "source_filename": meta.get("source_filename", "unknown"),
                "category": meta.get("category"),
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "excerpt": excerpt,
            })
        return sources

    def generate_answer(self, question: str, docs: List[Document]) -> Tuple[str, Optional[str], List[dict]]:
        """Returns (answer_text, ticket_id_or_none, sources)."""
        if not docs:
            ticket = create_ticket(question, reason="no_relevant_documents")
            answer = (
                f"FALLBACK_TRIGGERED: No relevant documents met the retrieval threshold. "
                f"A ticket has been created for follow-up: {ticket['ticket_id']}"
            )
            return answer, ticket["ticket_id"], []

        sources = self._build_sources(docs)

        # Each chunk is numbered to match its entry in `sources`, so the
        # model CAN cite inline with [1], [2], etc. if it chooses to -
        # this isn't enforced/parsed, it's just encouraged, since the
        # structured `sources` list is returned regardless either way.
        context_blocks = []
        for src in sources:
            doc = docs[src["source_number"] - 1]
            context_blocks.append(
                f"[{src['source_number']}] (source: {src['source_filename']}, "
                f"chunk {src['chunk_index']})\n{doc.page_content}"
            )
        context = "\n\n".join(context_blocks)

        prompt = (
            f"You are a knowledge-base assistant. Answer the user's question using ONLY "
            f"the provided context below. Each context block is labeled with a number in "
            f"square brackets, e.g. [1]. Where it's natural to do so, cite the source "
            f"number(s) your answer draws from inline, like this: \"...as described in "
            f"[1].\" If the context does not contain enough information to answer "
            f"confidently, say so explicitly rather than guessing.\n\n"
            f"User Question: {question}\n\n"
            f"Retrieved Context:\n{context}"
        )
        response = self.llm.invoke(prompt)
        return response.content, None, sources

    def execute_fallback(self, question: str) -> Tuple[str, str, List[dict]]:
        """Returns (answer_text, ticket_id, sources)."""
        ticket = create_ticket(question, reason="routed_as_unrelated")
        answer = (
            f"FALLBACK_TRIGGERED: No matching information was found in the knowledge base "
            f"for this query. A ticket has been created for follow-up: {ticket['ticket_id']}"
        )
        return answer, ticket["ticket_id"], []

    def run(self, question: str) -> AgentState:
        route_choice = self.determine_route(question)

        if route_choice == "vectorstore":
            docs = self.retrieve_docs(question)
            generation, ticket_id, sources = self.generate_answer(question, docs)
        else:
            docs = []
            generation, ticket_id, sources = self.execute_fallback(question)

        return {
            "question": question,
            "documents": docs,
            "generation": generation,
            "route_choice": route_choice,
            "ticket_id": ticket_id,
            "sources": sources,
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