import azure.functions as func
import json
import os
import time
import hashlib
import logging
import random
from core.rag_agent import get_agent
from core.ticketing import get_ticket, list_tickets
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Filter, FieldCondition, MatchValue

bp = func.Blueprint()

@bp.route(route="query", methods=["POST"])
def query(req: func.HttpRequest) -> func.HttpResponse:
    """
    Ask a question against the knowledge base.

    Expected JSON body:
    {
        "question": "What does the document say about X?"
    }
    """
    logging.info("Processing inbound RAG query.")

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
            "answer": final_output.get("generation"),
            "ticket_id": final_output.get("ticket_id")
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


# Tune these if you find chunks are cutting off mid-concept (raise chunk_size)
# or retrieval is pulling in irrelevant padding (lower chunk_size).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# --- Embedding rate limiting ---
#
# Google's free-tier RPM/TPM limits aren't published as fixed numbers in
# their docs (they depend on your account's usage tier) and change over
# time. Check your actual current limit at https://aistudio.google.com/rate-limit
# and set these two values in local.settings.json's Values to match:
#
#   "EMBEDDING_BATCH_SIZE": "10"             -> chunks embedded per API call
#   "EMBEDDING_REQUESTS_PER_MINUTE": "15"    -> your account's RPM limit for the model
#
# A large document gets split into many small batches, with a sleep
# between each one, so a single add_document call never exceeds your
# configured RPM regardless of how many chunks it produces.
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_RPM = 15


def _get_rate_limit_config():
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE))
    rpm = int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", DEFAULT_EMBEDDING_RPM))
    return max(1, batch_size), max(1, rpm)

def _add_documents_rate_limited(vector_store, docs: list[Document]) -> list:
    """
    Adds documents to the vector store in small batches, pacing requests 
    to stay under the RPM limit, and implementing exponential backoff 
    retries to gracefully handle any unexpected 429 / ResourceExhausted errors.
    """
    batch_size, rpm = _get_rate_limit_config()
    sleep_seconds = 60.0 / rpm

    all_ids = []
    total_batches = (len(docs) + batch_size - 1) // batch_size
    max_retries = 5

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch = docs[start:end]

        logging.info(
            f"Embedding batch {batch_num + 1}/{total_batches} "
            f"({len(batch)} chunk(s)) - pacing at {rpm} req/min"
        )

        batch_ids = None
        for attempt in range(max_retries):
            try:
                batch_ids = vector_store.add_documents(batch)
                break
            except Exception as e:
                error_str = str(e).lower()
                # Catch 429 / Resource Exhausted / Rate limit errors from Google API
                if "429" in error_str or "resource exhausted" in error_str or "rate limit" in error_str:
                    if attempt == max_retries - 1:
                        logging.error(f"Max retries reached for batch {batch_num + 1}. Failing.")
                        raise e
                    
                    # Exponential backoff with random jitter (e.g., 2s, 4s, 8s + jitter)
                    backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logging.warning(
                        f"Rate limit hit on batch {batch_num + 1}. "
                        f"Retrying in {backoff:.2f} seconds (Attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(backoff)
                else:
                    # Raise non-rate-limit errors immediately
                    raise e

        if batch_ids:
            all_ids.extend(batch_ids)

        # Don't sleep after the very last batch
        is_last_batch = (batch_num == total_batches - 1)
        if not is_last_batch:
            time.sleep(sleep_seconds)

    return all_ids

# def _add_documents_rate_limited(vector_store, docs: list[Document]) -> list:
#     """
#     Adds documents to the vector store in small batches, sleeping between
#     each one so the embedding API call rate stays under the configured
#     requests-per-minute limit. Each batch is one embedding API call.
#     """
#     batch_size, rpm = _get_rate_limit_config()
#     sleep_seconds = 60.0 / rpm

#     all_ids = []
#     total_batches = (len(docs) + batch_size - 1) // batch_size

#     for batch_num in range(total_batches):
#         start = batch_num * batch_size
#         end = start + batch_size
#         batch = docs[start:end]

#         logging.info(
#             f"Embedding batch {batch_num + 1}/{total_batches} "
#             f"({len(batch)} chunk(s)) - pacing at {rpm} req/min"
#         )

#         batch_ids = vector_store.add_documents(batch)
#         all_ids.extend(batch_ids)

#         # Don't sleep after the very last batch - nothing more to wait for.
#         is_last_batch = (batch_num == total_batches - 1)
#         if not is_last_batch:
#             time.sleep(sleep_seconds)

#     return all_ids


# --- Duplicate detection ---
#
# Computed on the FULL document text, before chunking. Whitespace is
# normalized (collapsed/trimmed) first so trivial differences - like a PDF
# extractor producing slightly different line breaks on a second pass of
# the same file - don't produce a different hash for genuinely identical
# content. Every chunk belonging to a document carries the same
# content_hash, so a single scroll-filter check tells us whether this
# exact document was already ingested, before any embedding calls happen.
def _compute_content_hash(page_content: str) -> str:
    normalized = " ".join(page_content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _find_existing_by_hash(agent, collection_name: str, content_hash: str):
    existing, _ = agent.client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.content_hash",
                    match=MatchValue(value=content_hash)
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return existing[0] if existing else None


@bp.route(route="documents", methods=["POST"])
def add_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    Adds a new document to the vector DB. The document text is automatically
    split into overlapping chunks before embedding/storage, so this single
    endpoint is safe to call with either a short snippet or a full multi-page
    document's extracted text - chunking happens here either way.

    Expected JSON body:
    {
        "page_content": "The full text content to chunk, embed, and store.",
        "metadata": {
            "category": "Research",
            "source_filename": "paper.pdf"   # optional but recommended
        }
    }
    `metadata` is optional. Any keys you pass are preserved on every chunk,
    plus this endpoint adds `chunk_index` and `total_chunks` automatically.
    """
    logging.info("Adding a new document to vector store.")
    try:
        req_body = req.get_json()
        page_content = req_body.get("page_content")
        metadata = req_body.get("metadata", {}) or {}

        if not page_content or not isinstance(page_content, str):
            return func.HttpResponse(
                json.dumps({"error": "Missing or invalid required field 'page_content' (must be a string)"}),
                status_code=400,
                mimetype="application/json"
            )

        agent = get_agent()
        collection_name = agent.vector_store.collection_name
        content_hash = _compute_content_hash(page_content)

        existing_point = _find_existing_by_hash(agent, collection_name, content_hash)
        if existing_point is not None:
            existing_metadata = existing_point.payload.get("metadata", {})
            logging.info(f"Duplicate content detected (hash={content_hash[:12]}...) - skipping re-ingestion.")
            return func.HttpResponse(
                json.dumps({
                    "status": "Skipped - duplicate content",
                    "content_hash": content_hash,
                    "matched_existing_source": existing_metadata.get("source_filename"),
                    "message": "This exact document content already exists in the knowledge base. No chunks were added."
                }),
                status_code=200,
                mimetype="application/json"
            )

        chunks = _text_splitter.split_text(page_content)

        if not chunks:
            return func.HttpResponse(
                json.dumps({"error": "page_content produced zero chunks after splitting (empty/whitespace-only text?)"}),
                status_code=400,
                mimetype="application/json"
            )

        total_chunks = len(chunks)
        new_docs = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = total_chunks
            chunk_metadata["content_hash"] = content_hash
            new_docs.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        added_ids = _add_documents_rate_limited(agent.vector_store, new_docs)

        return func.HttpResponse(
            json.dumps({
                "status": "Success",
                "total_chunks_added": total_chunks,
                "added_ids": [str(i) for i in added_ids],
                "content_hash": content_hash,
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


@bp.route(route="documents/by-source", methods=["DELETE"])
def delete_document_by_source(req: func.HttpRequest) -> func.HttpResponse:
    """
    Deletes every chunk that came from a given source file.

    Since add_document splits one uploaded file into many chunks (each
    stored as a separate point with its own id), deleting by a single
    point id only removes one chunk and leaves the rest orphaned. This
    route instead deletes by the `source_filename` metadata field that
    ingest_from_blob.py attaches to every chunk, so re-ingesting or
    removing a file is a single call regardless of how many chunks it
    produced.

    Usage:
        DELETE /api/documents/by-source?filename=paper.pdf
    """
    logging.info("Deleting document chunks by source_filename.")
    try:
        filename = req.params.get("filename")

        if not filename:
            return func.HttpResponse(
                json.dumps({"error": "Missing required query parameter 'filename'"}),
                status_code=400,
                mimetype="application/json"
            )

        agent = get_agent()
        collection_name = agent.vector_store.collection_name

        # metadata is stored nested under the "metadata" payload key by
        # QdrantVectorStore, so the filter path must match that nesting.
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source_filename",
                    match=MatchValue(value=filename)
                )
            ]
        )

        # Count matches first so we can report something meaningful -
        # Qdrant's delete call doesn't return how many points it removed.
        matching_points, _ = agent.client.scroll(
            collection_name=collection_name,
            scroll_filter=delete_filter,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        matched_count = len(matching_points)

        if matched_count == 0:
            return func.HttpResponse(
                json.dumps({"status": "No matching chunks found", "filename": filename, "deleted_count": 0}),
                status_code=404,
                mimetype="application/json"
            )

        agent.client.delete(
            collection_name=collection_name,
            points_selector=delete_filter,
        )

        return func.HttpResponse(
            json.dumps({
                "status": "Success",
                "filename": filename,
                "deleted_count": matched_count
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Failed to delete document by source: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to delete document"}),
            status_code=500,
            mimetype="application/json"
        )


@bp.route(route="tickets", methods=["GET"])
def view_tickets(req: func.HttpRequest) -> func.HttpResponse:
    """
    Lists every ticket created by the fallback flow, or looks up one
    specific ticket if ?ticket_id=... is passed. This is purely for local
    testing - a stand-in for logging into a real ServiceNow/Jira instance
    to confirm the escalation path actually fired and captured the
    right question.

    Usage:
        GET /api/tickets                      -> list all tickets
        GET /api/tickets?ticket_id=TKT-XXXXXX  -> view one ticket
    """
    logging.info("Viewing ticket(s).")
    try:
        ticket_id = req.params.get("ticket_id")

        if ticket_id:
            ticket = get_ticket(ticket_id)
            if ticket is None:
                return func.HttpResponse(
                    json.dumps({"error": f"No ticket found with id '{ticket_id}'"}),
                    status_code=404,
                    mimetype="application/json"
                )
            return func.HttpResponse(
                json.dumps(ticket),
                status_code=200,
                mimetype="application/json"
            )

        all_tickets = list_tickets()
        return func.HttpResponse(
            json.dumps({"count": len(all_tickets), "tickets": all_tickets}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Failed to retrieve ticket(s): {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Failed to retrieve ticket(s)"}),
            status_code=500,
            mimetype="application/json"
        )