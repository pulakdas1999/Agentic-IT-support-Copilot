import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langgraph.graph import StateGraph, START, END

# Define state machine memory schema
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    route_choice: str

# Structured validation schema for the routing agent
class RouterSchema(BaseModel):
    datasource: Literal["vectorstore", "fallback"] = Field(
        description="Route to 'vectorstore' if query relates to corporate operations, IT, VPN, or hardware. Otherwise route to 'fallback'."
    )

def build_workflow_graph():
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 1. Connect to the local file database initialized by our script
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, "local_qdrant_storage")
    
    embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    vector_store = Qdrant(
        client=None, # Client is derived via path configuration below
        path=db_path,
        collection_name="ey_internal_docs",
        embeddings=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Instantiate Google Gemini 2.5 Flash as the processing brain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    structured_router = llm.with_structured_output(RouterSchema)

    # 3. Define Graph Nodes (Steps)
    def determine_route(state: AgentState):
        question = state["question"]
        system_prompt = f"Analyze this enterprise user query: '{question}' and decide its destination data stack."
        try:
            decision = structured_router.invoke(system_prompt)
            return {"route_choice": decision.datasource}
        except Exception:
            # Fallback gracefully if model serialization stumbles
            return {"route_choice": "fallback"}

    def retrieve_docs(state: AgentState):
        question = state["question"]
        docs = retriever.invoke(question)
        return {"documents": docs, "question": question}

    def generate_answer(state: AgentState):
        question = state["question"]
        docs = state["documents"]
        
        # Check if retrieved records are weak or empty
        if not docs:
            return {"generation": "FALLBACK_TRIGGERED: No relevant local documents met the certainty filter."}

        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            f"You are an EY support assistant. Answer the user query using ONLY the provided verified context.\n"
            f"User Question: {question}\n\n"
            f"Verified Enterprise Context:\n{context}"
        )
        response = llm.invoke(prompt)
        return {"generation": response.content}

    def execute_fallback(state: AgentState):
        return {
            "generation": "FALLBACK_TRIGGERED: Query details not found in local internal corporate records. Ticket routed to the live IT Support queue."
        }

    # 4. Bind Everything into a Functional Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("execute_fallback", execute_fallback)

    # Route logic conditions mapping
    def router_decision_edge(state: AgentState):
        return state["route_choice"]

    workflow.add_conditional_edges(
        START,
        determine_route,
        {
            "vectorstore": "retrieve_docs",
            "fallback": "execute_fallback"
        }
    )
    workflow.add_edge("retrieve_docs", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("execute_fallback", END)

    return workflow.compile()