"""
database/nlu_logs.py

Lightweight helpers to record NLU interactions (queries, predicted intents,
confidence scores, extracted entities, session/account and timestamps) into
SQLite so the app can show analytics and export logs.
"""
from typing import List, Dict, Any, Optional
import sqlite3
import json
from datetime import datetime
from .db import get_conn


def init_logs_table() -> None:
    """Create the nlu_logs table if it doesn't exist."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nlu_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            account_number TEXT,
            query TEXT,
            bot_response TEXT,
            predicted_intents TEXT,
            top_intent TEXT,
            top_confidence REAL,
            entities TEXT,
            success INTEGER DEFAULT 1,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    
    # Add bot_response and success columns if they don't exist
    try:
        cur.execute("ALTER TABLE nlu_logs ADD COLUMN bot_response TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cur.execute("ALTER TABLE nlu_logs ADD COLUMN success INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.commit()
    conn.close()


def log_interaction(
    session_id: str,
    account_number: Optional[str],
    query: str,
    predicted_intents: List[tuple],
    entities: Dict[str, Any],
    bot_response: str = "",
) -> None:
    """Insert a log record for a user query.

    - `predicted_intents` is a list of (label, score) pairs.
    - `entities` is a dictionary or list-serialisable object.
    - `bot_response` is the response from the bot
    """
    conn = get_conn()
    cur = conn.cursor()

    top_intent = None
    top_conf = None
    if predicted_intents:
        top_intent = predicted_intents[0][0]
        try:
            top_conf = float(predicted_intents[0][1])
        except Exception:
            top_conf = None

    cur.execute(
        "INSERT INTO nlu_logs (session_id, account_number, query, bot_response, predicted_intents, top_intent, top_confidence, entities, success, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            account_number,
            query,
            bot_response,
            json.dumps(predicted_intents, ensure_ascii=False),
            top_intent,
            top_conf,
            json.dumps(entities, ensure_ascii=False),
            1,  # success
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def log_llm_interaction(
    session_id: str,
    account_number: Optional[str],
    query: str,
    bot_response: str = "",
) -> None:
    """Log a non-banking query handled by LLM.
    
    For LLM interactions, we log with:
    - top_intent = "llm_response" (special marker)
    - top_confidence = 1.0 (always confident for LLM)
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO nlu_logs (session_id, account_number, query, bot_response, predicted_intents, top_intent, top_confidence, entities, success, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            account_number,
            query,
            bot_response,
            json.dumps([("llm_response", 1.0)], ensure_ascii=False),
            "llm_response",
            1.0,
            json.dumps({}, ensure_ascii=False),
            1,  # success
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent logs (both banking and LLM queries)."""
    conn = get_conn()
    cur = conn.cursor()
    # Show all queries: both banking (top_intent != 'llm_response') and LLM (top_intent = 'llm_response')
    cur.execute(
        "SELECT id, session_id, account_number, query, bot_response, predicted_intents, top_intent, top_confidence, entities, success, timestamp FROM nlu_logs ORDER BY id DESC LIMIT ?", 
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "session_id": r[1],
            "account_number": r[2],
            "query": r[3],
            "bot_response": r[4] or "",
            "predicted_intents": json.loads(r[5]) if r[5] else [],
            "top_intent": r[6],
            "top_confidence": r[7],
            "entities": json.loads(r[8]) if r[8] else {},
            "success": r[9] if len(r) > 9 else 1,
            "timestamp": r[10],
        })
    return results


def get_intent_distribution() -> List[Dict[str, Any]]:
    """Get intent distribution for banking queries only."""
    conn = get_conn()
    cur = conn.cursor()
    # Only count queries where we actually ran NLU
    cur.execute(
        "SELECT top_intent, COUNT(*) as cnt FROM nlu_logs WHERE top_intent IS NOT NULL GROUP BY top_intent ORDER BY cnt DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [{"intent": r[0] or "<none>", "count": r[1]} for r in rows]


def get_confidence_stats() -> List[float]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT top_confidence FROM nlu_logs WHERE top_confidence IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return [float(r[0]) for r in rows]


def export_logs_as_csv() -> str:
    """Return CSV string of recent logs (suitable for download)."""
    import csv
    from io import StringIO

    logs = get_recent_logs(1000)
    if not logs:
        return ""

    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "timestamp", "session_id", "account_number", "query", "top_intent", "top_confidence", "predicted_intents", "entities"])
    for l in logs:
        writer.writerow([
            l.get("id"),
            l.get("timestamp"),
            l.get("session_id"),
            l.get("account_number"),
            l.get("query"),
            l.get("top_intent"),
            l.get("top_confidence"),
            json.dumps(l.get("predicted_intents"), ensure_ascii=False),
            json.dumps(l.get("entities"), ensure_ascii=False),
        ])

    return out.getvalue()


def clear_nlu_logs() -> None:
    """Clear all NLU logs (used for resetting analytics)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM nlu_logs")
    conn.commit()
    conn.close()
