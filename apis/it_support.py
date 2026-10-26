import azure.functions as func
import json
import os
import logging
from core.graph import build_workflow_graph

# Instantiate an independent blueprint module
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

        # Verify local database directory presence
        base_dir = os.path.dirname(os.path.dirname(__file__)) # Adjusted up one step due to subfolder
        db_path = os.path.join(base_dir, "local_qdrant_storage")
        if not os.path.exists(db_path):
            return func.HttpResponse(
                json.dumps({"error": "Local Database folder missing. Run 'python scripts/init_db.py' first!"}),
                status_code=500,
                mimetype="application/json"
            )

        # Run the LangGraph execution flow using Gemini
        agent_graph = build_workflow_graph()
        initial_state = {"question": user_question, "documents": [], "generation": "", "route_choice": ""}
        final_output = agent_graph.invoke(initial_state)

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