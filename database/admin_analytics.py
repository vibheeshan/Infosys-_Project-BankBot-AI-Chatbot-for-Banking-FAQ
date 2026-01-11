"""
database/admin_analytics.py

Analytics and metrics functions for the Admin Panel.
Provides calculated metrics, charts data, and filtering functions.
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from . import nlu_logs


def get_dashboard_metrics() -> Dict[str, Any]:
    """
    Get top-level metrics for dashboard cards.
    
    Returns:
        dict with keys: total_queries, total_intents, total_entities, 
                       success_rate, low_confidence_count, avg_confidence
    """
    try:
        logs = nlu_logs.get_recent_logs(limit=100000)
    except Exception:
        logs = []
    
    if not logs:
        return {
            "total_queries": 0,
            "total_intents": 0,
            "total_entities": 0,
            "success_rate": 0.0,
            "low_confidence_count": 0,
            "avg_confidence": 0.0,
        }
    
    total = len(logs)
    
    # Unique intents
    intents = set()
    for log in logs:
        if log.get("top_intent"):
            intents.add(log["top_intent"])
    
    # Total entities extracted
    entity_count = 0
    for log in logs:
        entities = log.get("entities", {})
        if isinstance(entities, dict):
            entity_count += len(entities)
        elif isinstance(entities, list):
            entity_count += len(entities)
    
    # Success rate: assume success if confidence > 0.5
    success_count = 0
    for log in logs:
        conf = log.get("top_confidence", 0)
        if conf and float(conf) > 0.5:
            success_count += 1
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    # Low confidence count (< 0.8)
    low_conf_count = sum(1 for log in logs if log.get("top_confidence", 1) and float(log["top_confidence"]) < 0.8)
    
    # Average confidence
    confs = []
    for log in logs:
        c = log.get("top_confidence")
        if c:
            try:
                confs.append(float(c))
            except Exception:
                pass
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    
    return {
        "total_queries": total,
        "total_intents": len(intents),
        "total_entities": entity_count,
        "success_rate": round(success_rate, 1),
        "low_confidence_count": low_conf_count,
        "avg_confidence": round(avg_conf, 2),
    }


def get_intent_usage_breakdown() -> List[Dict[str, Any]]:
    """
    Get intent usage breakdown with percentages.
    
    Returns:
        list of dicts: [{"intent": "...", "count": N, "percentage": X.X}, ...]
    """
    try:
        dist = nlu_logs.get_intent_distribution()
    except Exception:
        dist = []
    
    if not dist:
        return []
    
    total = sum(item.get("count", 0) for item in dist)
    result = []
    
    for item in dist:
        intent = item.get("intent", "<unknown>")
        count = item.get("count", 0)
        pct = (count / total * 100) if total > 0 else 0
        result.append({
            "intent": intent,
            "count": count,
            "percentage": round(pct, 1),
        })
    
    return sorted(result, key=lambda x: x["count"], reverse=True)


def get_confidence_distribution_data() -> Tuple[List[float], List[int]]:
    """
    Get confidence distribution for histogram.
    
    Returns:
        tuple: (bin_centers, counts) for plotting
    """
    try:
        confs = nlu_logs.get_confidence_stats()
    except Exception:
        confs = []
    
    if not confs:
        return [], []
    
    # Create bins: 0.0-0.2, 0.2-0.4, ..., 0.8-1.0
    import numpy as np
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(confs, bins=bins)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    
    return bin_centers, list(counts)


def get_daily_query_count() -> int:
    """Get count of queries from today."""
    try:
        logs = nlu_logs.get_recent_logs(limit=100000)
    except Exception:
        logs = []
    
    today = datetime.utcnow().date()
    count = 0
    for log in logs:
        ts = log.get("timestamp", "")
        try:
            log_date = datetime.fromisoformat(ts).date()
            if log_date == today:
                count += 1
        except Exception:
            pass
    
    return count


def get_queries_by_account() -> List[Dict[str, Any]]:
    """Get query count grouped by account number."""
    try:
        logs = nlu_logs.get_recent_logs(limit=100000)
    except Exception:
        logs = []
    
    account_map = {}
    for log in logs:
        acc = log.get("account_number") or "Unknown"
        account_map[acc] = account_map.get(acc, 0) + 1
    
    return [{"account": k, "count": v} for k, v in sorted(account_map.items(), key=lambda x: x[1], reverse=True)]


def get_most_used_intents(limit: int = 5) -> List[Dict[str, Any]]:
    """Get top N most used intents."""
    breakdown = get_intent_usage_breakdown()
    return breakdown[:limit]


def get_low_confidence_queries(threshold: float = 0.8, limit: int = 10) -> List[Dict[str, Any]]:
    """Get queries with confidence below threshold."""
    try:
        logs = nlu_logs.get_recent_logs(limit=100000)
    except Exception:
        logs = []
    
    low_conf = [log for log in logs if log.get("top_confidence", 1) and float(log["top_confidence"]) < threshold]
    return sorted(low_conf, key=lambda x: x.get("top_confidence", 0))[:limit]


def get_session_stats() -> Dict[str, Any]:
    """Get statistics per session."""
    try:
        logs = nlu_logs.get_recent_logs(limit=100000)
    except Exception:
        logs = []
    
    session_map = {}
    for log in logs:
        sid = log.get("session_id", "unknown")
        if sid not in session_map:
            session_map[sid] = {"queries": 0, "avg_confidence": [], "intents": []}
        session_map[sid]["queries"] += 1
        conf = log.get("top_confidence")
        if conf:
            try:
                session_map[sid]["avg_confidence"].append(float(conf))
            except Exception:
                pass
        intent = log.get("top_intent")
        if intent:
            session_map[sid]["intents"].append(intent)
    
    result = []
    for sid, data in session_map.items():
        avg_conf = sum(data["avg_confidence"]) / len(data["avg_confidence"]) if data["avg_confidence"] else 0
        intents = set(data["intents"])
        result.append({
            "session_id": sid,
            "query_count": data["queries"],
            "avg_confidence": round(avg_conf, 2),
            "unique_intents": len(intents),
        })
    
    return sorted(result, key=lambda x: x["query_count"], reverse=True)
