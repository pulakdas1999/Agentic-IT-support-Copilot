import azure.functions as func
import json
import os
import logging
from core.graph import get_agent
from langchain_core.documents import Document

bp = func.Blueprint()

@bp.route(route="it_support_agent", methods=["POST"])
def it_support_agent(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing inbound Agentic RAG request via Blueprint router.")

    try:
        req_body = req.get_json()
        user_question = req_body.get("question")

        if not user_question:
            return func.HttpResponse(
                json.dumps({"error": "Missing required field 'question' in JSON body"}),
                status_code=400,
                mimetype="application/json"
            )

        base_dir = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, "local_qdrant_storage")
        if not os.path.exists(db_path):
            return func.HttpResponse(
                json.dumps({"error": "Local Database folder missing. Run 'python scripts/init_db.py' first!"}),
                status_code=500,
                mimetype="application/json"
            )

        agent = get_agent()
        final_output = agent.run(user_question)

        response_payload = {
            "status": "Success",
            "routed_through": final_output.get("route_choice", "direct_step"),
            "documents_matched": len(final_output.get("documents", [])),
            "answer": final_output.get("generation")
        }

        return func.HttpResponse(
            json.dumps(response_payload),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Blueprint execution failure: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal pipeline failure during blueprint execution"}),
            status_code=500,
            mimetype="application/json"
        )

@bp.route(route="documents", methods=["GET"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Listing documents from vector store.")
    try:
        agent = get_agent()
        collection_name = agent.vector_store.collection_name

        count = agent.client.count(collection_name=collection_name).count

        points, _ = agent.client.scroll(
            collection_name=collection_name,
            limit=count or 100,
            with_payload=True,
            with_vectors=False,
        )

        documents = [
            {
                "id": p.id,
                "page_content": p.payload.get("page_content"),
                "metadata": p.payload.get("metadata"),
            }
            for p in points
        ]

        return func.HttpResponse(
            json.dumps({"count": len(documents), "documents": documents}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Failed to list documents: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to retrieve documents"}),
            status_code=500,
            mimetype="application/json"
        )


@bp.route(route="documents", methods=["POST"])
def add_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    Adds a new document to the vector DB.

    Expected JSON body:
    {
        "page_content": "The text content to embed and store.",
        "metadata": {"category": "IT_Infrastructure", "severity": "Low"}
    }
    `metadata` is optional.
    """
    logging.info("Adding a new document to vector store.")
    try:
        req_body = req.get_json()
        page_content = req_body.get("page_content")
        metadata = req_body.get("metadata", {})

        if not page_content or not isinstance(page_content, str):
            return func.HttpResponse(
                json.dumps({"error": "Missing or invalid required field 'page_content' (must be a string)"}),
                status_code=400,
                mimetype="application/json"
            )

        agent = get_agent()
        new_doc = Document(page_content=page_content, metadata=metadata)

        added_ids = agent.vector_store.add_documents([new_doc])

        return func.HttpResponse(
            json.dumps({
                "status": "Success",
                "added_id": str(added_ids[0]) if added_ids else None,
                "page_content": page_content,
                "metadata": metadata
            }),
            status_code=201,
            mimetype="application/json"
        )

    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON"}),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Failed to add document: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to add document"}),
            status_code=500,
            mimetype="application/json"
        )