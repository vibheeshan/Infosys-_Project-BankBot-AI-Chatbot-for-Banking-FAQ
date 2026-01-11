# nlu_engine/intent_classifier.py

def classify_intent(text: str) -> str:
    """Classify intent from text and return intent label.

    This function is keyword-based (easy to extend to ML later).
    Numeric-only inputs should not be mapped to an intent here.
    """
    if not text or not isinstance(text, str):
        return "fallback"

    text_l = text.lower().strip()
    # do not infer intent from pure numbers
    if text_l.isdigit() or text_l.replace(".", "").replace(",", "").isdigit():
        return "fallback"

    # --- Reject non-banking "transfer ..." patterns early ---
    non_banking_transfer_patterns = [
        "transfer llm",
        "transfer model",
        "transfer file",
        "transfer code",
        "transfer data",
        "transfer learning",
        "transfer knowledge",
        "transfer document",
        "transfer project",
    ]
    if any(p in text_l for p in non_banking_transfer_patterns):
        return "fallback"
    # --- end added check ---

    # ATM / branch phrases (explicit checks first)
    atm_phrases = [
        "find atm",
        "nearby atm",
        "atm near me",
        "atm nearby",
        "branch details",
        "branch near",
        "nearest atm",
    ]
    if any(p in text_l for p in atm_phrases) or any(k in text_l for k in ["atm", "branch", "nearest", "location", "address"]):
        return "find_atm"
    if "balance" in text_l or "check balance" in text_l:
        return "check_balance"
    if "transfer" in text_l or "send" in text_l or "pay" in text_l:
        return "transfer_money"
    if "block" in text_l and "card" in text_l or "block card" in text_l:
        return "block_card"
    if "loan" in text_l:
        return "loan_info"

    return "fallback"


def classify(text: str) -> tuple:
    """Classify intent and return (intent, confidence) tuple.
    
    Args:
        text: User input text
        
    Returns:
        tuple: (intent: str, confidence: float)
    """
    if not text or not isinstance(text, str):
        return ("fallback", 0.0)
    
    intent = classify_intent(text)
    
    # Higher confidence for explicit keywords, lower for fallback
    text_lower = text.lower()
    if intent == "find_atm":
        confidence = 0.95
    elif intent == "check_balance" and "balance" in text_lower:
        confidence = 0.9
    elif intent == "transfer_money" and ("transfer" in text_lower or "send" in text_lower):
        confidence = 0.9
    elif intent == "block_card" and "block" in text_lower:
        confidence = 0.9
    elif intent == "loan_info" and "loan" in text_lower:
        confidence = 0.9
    else:
        confidence = 0.5
    
    return (intent, confidence)
