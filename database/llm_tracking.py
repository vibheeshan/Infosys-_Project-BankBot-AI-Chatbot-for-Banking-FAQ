"""
database/llm_tracking.py

Track LLM fallback usage and performance metrics.
Logs when NLU confidence is low and LLM is used as fallback.
"""

from typing import List, Dict, Any, Optional
import sqlite3
import json
from datetime import datetime
from .db import get_conn


def init_llm_tracking_table() -> None:
    """Create the llm_usage table if it doesn't exist."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            account_number TEXT,
            query TEXT,
            nlu_intent TEXT,
            nlu_confidence REAL,
            fallback_reason TEXT,
            llm_response TEXT,
            llm_model TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def log_llm_usage(
    session_id: str,
    account_number: Optional[str],
    query: str,
    nlu_intent: Optional[str],
    nlu_confidence: float,
    fallback_reason: str,
    llm_response: str,
    llm_model: str = "groq",
) -> None:
    """
    Log when LLM is used as fallback.
    
    Args:
        session_id: User session ID
        account_number: User's account number
        query: Original user query
        nlu_intent: Detected intent (if any)
        nlu_confidence: NLU confidence score
        fallback_reason: Why LLM was used (e.g., "low_confidence", "unknown_intent")
        llm_response: Response from LLM
        llm_model: Which LLM model was used
    """
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute(
        """
        INSERT INTO llm_usage 
        (session_id, account_number, query, nlu_intent, nlu_confidence, 
         fallback_reason, llm_response, llm_model, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            account_number,
            query,
            nlu_intent,
            float(nlu_confidence),
            fallback_reason,
            llm_response,
            llm_model,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_llm_usage(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent LLM fallback usage logs."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM llm_usage ORDER BY id DESC LIMIT ?",
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
            "nlu_intent": r[4],
            "nlu_confidence": r[5],
            "fallback_reason": r[6],
            "llm_response": r[7],
            "llm_model": r[8],
            "timestamp": r[9],
        })
    return results


def get_llm_fallback_count() -> int:
    """Get total count of LLM fallback usages."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM llm_usage")
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_fallback_reasons_breakdown() -> List[Dict[str, Any]]:
    """Get breakdown of why LLM was used (fallback reasons)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fallback_reason, COUNT(*) as cnt 
        FROM llm_usage 
        GROUP BY fallback_reason 
        ORDER BY cnt DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [{"reason": r[0], "count": r[1]} for r in rows]


def get_llm_models_usage() -> List[Dict[str, Any]]:
    """Get breakdown of which LLM models were used."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT llm_model, COUNT(*) as cnt 
        FROM llm_usage 
        GROUP BY llm_model 
        ORDER BY cnt DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [{"model": r[0], "count": r[1]} for r in rows]


def get_average_nlu_confidence_of_fallbacks() -> float:
    """Get average NLU confidence for queries that fell back to LLM."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT AVG(nlu_confidence) FROM llm_usage WHERE nlu_confidence IS NOT NULL"
    )
    result = cur.fetchone()[0]
    conn.close()
    return round(float(result), 2) if result else 0.0


def get_llm_usage_over_time(days: int = 7) -> List[Dict[str, Any]]:
    """Get LLM usage count per day for the last N days."""
    from datetime import datetime, timedelta
    
    conn = get_conn()
    cur = conn.cursor()
    
    # Get logs from last N days
    cur.execute(
        """
        SELECT DATE(timestamp) as day, COUNT(*) as count
        FROM llm_usage
        WHERE timestamp >= datetime('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
        """,
        (f"-{days} days",),
    )
    rows = cur.fetchall()
    conn.close()
    
    return [{"date": r[0], "count": r[1]} for r in rows]


def get_today_llm_count() -> int:
    """Get count of LLM usages today."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM llm_usage 
        WHERE DATE(timestamp) = DATE('now')
        """
    )
    count = cur.fetchone()[0]
    conn.close()
    return count


def export_llm_usage_as_csv() -> str:
    """Export LLM usage logs as CSV."""
    import csv
    from io import StringIO
    
    logs = get_recent_llm_usage(1000)
    if not logs:
        return ""
    
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "id", "timestamp", "session_id", "account_number", "query",
        "nlu_intent", "nlu_confidence", "fallback_reason", "llm_model", "llm_response"
    ])
    
    for log in logs:
        writer.writerow([
            log.get("id"),
            log.get("timestamp"),
            log.get("session_id"),
            log.get("account_number"),
            log.get("query"),
            log.get("nlu_intent"),
            log.get("nlu_confidence"),
            log.get("fallback_reason"),
            log.get("llm_model"),
            log.get("llm_response"),
        ])
    
    return out.getvalue()
