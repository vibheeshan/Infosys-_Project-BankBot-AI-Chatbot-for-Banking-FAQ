"""
NLU Engine for BankBot AI.
Handles intent recognition, entity extraction, and NLU routing.
"""

from nlu_engine.infer_intent import predict_intent
from nlu_engine.entity_extractor import extract_entities
from nlu_engine.nlu_router import process_query

__all__ = [
    "predict_intent",
    "extract_entities",
    "process_query",
]
