import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "audit.jsonl"

_log_lock = threading.Lock()


def ensure_log_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_query_event(
    user_id: str,
    user_groups: list[str],
    question: str,
    answer: str,
    retrieval_audit: dict[str, Any],
    trace_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> dict[str, Any]:
    """
    Log a structured enterprise access & query event to logs/audit.jsonl.
    """
    ensure_log_dir()

    authorized_count = retrieval_audit.get("authorized_count", 0)
    denied_count = retrieval_audit.get("denied_count", 0)

    if authorized_count > 0 and denied_count == 0:
        access_decision = "GRANTED"
    elif authorized_count > 0 and denied_count > 0:
        access_decision = "PARTIAL_ACCESS"
    elif authorized_count == 0 and denied_count > 0:
        access_decision = "DENIED"
    else:
        access_decision = "NO_MATCH"

    refusal_phrase = "I couldn't find relevant information in the available knowledge base."
    is_refusal = refusal_phrase.lower() in answer.lower()

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id or "",
        "user_id": user_id,
        "user_groups": list(user_groups),
        "question": question,
        "access_decision": access_decision,
        "is_refusal": is_refusal,
        "retrieval": {
            "candidates_scanned": retrieval_audit.get("candidates_scanned", 0),
            "authorized_count": authorized_count,
            "denied_count": denied_count,
            "authorized_sources": retrieval_audit.get("authorized_sources", []),
            "denied_sources": retrieval_audit.get("denied_sources", []),
        },
        "answer_preview": answer[:150] + ("..." if len(answer) > 150 else ""),
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
    }

    with _log_lock:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    return event


def read_audit_logs(limit: int = 100) -> list[dict[str, Any]]:
    """Read the latest audit logs from logs/audit.jsonl."""
    if not AUDIT_LOG_FILE.exists():
        return []

    records = []
    with _log_lock:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records[-limit:]


def get_observability_summary() -> dict[str, Any]:
    """Compute aggregate observability metrics across recorded audit logs."""
    logs = read_audit_logs(limit=1000)
    if not logs:
        return {
            "total_queries": 0,
            "access_denial_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_latency_ms": 0.0,
            "top_authorized_docs": [],
            "top_denied_docs": [],
            "user_activity": {},
        }

    total = len(logs)
    denied = sum(1 for log in logs if log.get("access_decision") == "DENIED")
    refusals = sum(1 for log in logs if log.get("is_refusal"))
    
    latencies = [log["latency_ms"] for log in logs if log.get("latency_ms") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    auth_docs: dict[str, int] = {}
    denied_docs: dict[str, int] = {}
    user_activity: dict[str, int] = {}

    for log in logs:
        uid = log.get("user_id", "unknown")
        user_activity[uid] = user_activity.get(uid, 0) + 1

        retrieval = log.get("retrieval", {})
        for src in retrieval.get("authorized_sources", []):
            auth_docs[src] = auth_docs.get(src, 0) + 1

        for denied_entry in retrieval.get("denied_sources", []):
            src = denied_entry.get("source_path", "unknown")
            denied_docs[src] = denied_docs.get(src, 0) + 1

    sorted_auth = sorted(auth_docs.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_denied = sorted(denied_docs.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_queries": total,
        "access_denial_rate": round((denied / total) * 100, 1),
        "refusal_rate": round((refusals / total) * 100, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "top_authorized_docs": sorted_auth,
        "top_denied_docs": sorted_denied,
        "user_activity": user_activity,
    }

