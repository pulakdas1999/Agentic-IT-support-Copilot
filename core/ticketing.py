"""
A lightweight local mock ticketing system - stands in for a real platform
like ServiceNow or Jira so the escalation flow can be tested end-to-end
without external signups or network calls.

Tickets are stored as a single JSON file under local_ticket_storage/,
alongside local_qdrant_storage/ at the project root. This is intentionally
simple (no real database) since its only job is local testing - swap this
module out for a real ServiceNow/Jira REST client later without touching
any of its callers, since they only depend on create_ticket/get_ticket/
list_tickets's signatures, not on how tickets are stored.
"""
import os
import json
import uuid
import threading
from datetime import datetime, timezone

# Guards read-modify-write of the JSON file against concurrent requests.
# Good enough for local single-process testing; a real ticketing backend
# would handle this itself.
_file_lock = threading.Lock()


def _get_tickets_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tickets_dir = os.path.join(base_dir, "local_ticket_storage")
    os.makedirs(tickets_dir, exist_ok=True)
    return os.path.join(tickets_dir, "tickets.json")


def _load_tickets() -> list:
    path = _get_tickets_path()
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_tickets(tickets: list) -> None:
    path = _get_tickets_path()
    with open(path, "w") as f:
        json.dump(tickets, f, indent=2)


def create_ticket(question: str, reason: str) -> dict:
    """
    Creates a new ticket for a question the knowledge base couldn't answer.

    reason: short machine-readable explanation, e.g.
            "no_relevant_documents" or "routed_as_unrelated"
    """
    with _file_lock:
        tickets = _load_tickets()

        ticket = {
            "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "question": question,
            "reason": reason,
            "status": "Open",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        tickets.append(ticket)
        _save_tickets(tickets)
        return ticket


def get_ticket(ticket_id: str) -> dict | None:
    tickets = _load_tickets()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            return t
    return None


def list_tickets() -> list:
    return _load_tickets()