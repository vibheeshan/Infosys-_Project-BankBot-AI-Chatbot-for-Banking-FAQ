"""
Domain Gate: Determines if a query is banking or non-banking.

This module acts as a gatekeeper before routing to either:
- Banking NLU + Dialogue Manager (for banking intents)
- Groq LLM (for general knowledge, greetings, etc.)
"""

from typing import Literal


# Banking keywords that MUST be handled by rule-based NLU
BANKING_KEYWORDS = {
    # Balance checks
    "balance", "check", "check balance",
    
    # Transfers
    "transfer", "send", "pay", "money",
    
    # Card operations
    "block", "card", "unblock", "debit", "credit",
    
    # ATM/Branch
    "atm", "branch", "nearest", "location", "address",
    
    # Loans
    "loan", "borrow", "credit", "interest", "rate", "mortgage",
    
     # Account operations
    "account", "account number", "transaction", "history",
}

# Non-banking keywords that should route to LLM
NON_BANKING_KEYWORDS = {
    # Greetings
    "hi", "hello", "hey", "good morning", "good afternoon",
    
    # General questions
    "what is", "what are", "how do", "how to", "explain", "tell me about",
    "who is", "who are", "where is", "when is", "why is",
    
    # Tech/General topics
    "python", "java", "javascript", "c programming", "machine learning",
    "deep learning", "artificial intelligence", "ai", "data scientist",
    "data science", "science", "algorithm", "database", "web", "api",
}

# Non-banking transfer patterns (should NOT be routed to banking)
NON_BANKING_TRANSFER_PATTERNS = {
    "transfer llm", "transfer model", "transfer file", "transfer code",
    "transfer data", "transfer function", "transfer learning",
    "transfer knowledge", "transfer information", "transfer document",
    "transfer project",
}


def is_numeric_only(text: str) -> bool:
    """Check if input is purely numeric (account number, PIN, amount)."""
    if not text:
        return False
    normalized = text.strip().replace(".", "").replace(",", "").replace("-", "")
    return normalized.isdigit()


def is_non_banking_transfer(text: str) -> bool:
    """Detect non-banking 'transfer' contexts (files, code, models, etc.)."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in NON_BANKING_TRANSFER_PATTERNS)


def is_greeting(text: str) -> bool:
    """Check if input is a greeting."""
    text_norm = text.lower().strip().rstrip("!?.")
    greetings = ["hi", "hello", "hey"]
    return any(text_norm == g or text_norm.startswith(g + " ") for g in greetings)


def get_domain(text: str) -> Literal["banking", "non_banking", "unknown"]:
    """
    Determine if query belongs to banking or non-banking domain.
    
    Args:
        text: User input text
        
    Returns:
        "banking" - Route to NLU + Dialogue Manager
        "non_banking" - Route to Groq LLM
        "unknown" - No clear domain (treat as banking to maintain conversation)
    
    CRITICAL RULES:
    1. Numeric-only inputs are NEVER non-banking (they fill slots)
    2. "transfer X" where X is non-banking is routed to LLM
    3. General knowledge question patterns ("what is", "how do", etc.) route to LLM
    4. Explicit banking keywords always route to NLU
    5. Explicit non-banking keywords route to LLM
    6. Ambiguous queries default to banking (preserve intent context)
    """
    
    if not text or not isinstance(text, str):
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # R Numeric-only inputs are NEVER non-banking
    # They are slot fillers (account numbers, PINs, amounts)
    if is_numeric_only(text):
        return "banking"
    
    # Check for non-banking transfer patterns first
    if is_non_banking_transfer(text):
        return "non_banking"
    
    #  General knowledge question patterns route to LLM
    # These take precedence over banking keywords
    general_knowledge_patterns = [
        "what is", "what are", "how do", "how to", "how does",
        "explain", "tell me about", "who is", "who are",
        "where is", "when is", "why is", "what does"
    ]
    if any(pattern in text_lower for pattern in general_knowledge_patterns):
        return "non_banking"
    
    #  Explicit banking keywords route to banking
    if any(keyword in text_lower for keyword in BANKING_KEYWORDS):
        return "banking"
    
    #  Explicit non-banking keywords route to LLM
    if any(keyword in text_lower for keyword in NON_BANKING_KEYWORDS):
        return "non_banking"
    
    # Ambiguous queries default to banking (preserve conversation context)
    return "banking"


def should_reset_context(domain: str, current_intent: str) -> bool:
    """
    Decide if context should be reset when switching domains.
    
    Args:
        domain: New domain classification ("banking" or "non_banking")
        current_intent: Current intent in session context
        
    Returns:
        True if context should be reset, False otherwise
    """
    # Reset only when switching FROM banking TO non-banking
    # This prevents banking slots from leaking into LLM responses
    if domain == "non_banking" and current_intent in ["check_balance", "transfer_money", "block_card", "loan_info", "find_atm"]:
        return True
    
    # Keep context when staying in banking or when current_intent is None
    return False
