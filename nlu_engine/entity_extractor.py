import re
import spacy
from pathlib import Path

# Try to load the full spaCy English model, but fall back to a blank pipeline
# so the module does not crash at import time if the model isn't installed.
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print(
        "Warning: spaCy model 'en_core_web_sm' not found. Falling back to a blank 'en' pipeline.\n"
        "For better NER install it with: python -m spacy download en_core_web_sm"
    )
    nlp = spacy.blank("en")


def extract_entities(text: str):
    """Extract entities and return as list of entity dicts for nlu_router compatibility.
    
    Returns:
        list: List of entity dicts with format [{"label": "account_number", "value": "1001"}, ...]
    """
    doc = nlp(text)
    entities = {}
    entities_list = []

    # money amounts - parse into structured format (extract FIRST before accounts)
    money_parsed = []
    money_values = set()  # Track extracted money values to exclude from account numbers
    
    # Pattern 1: number followed by rupees/rs/₹
    pattern1_matches = re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:rupees|rupee|rs|rs\.|inr|INR|₹)\b", text, flags=re.IGNORECASE)
    for match in pattern1_matches:
        val = float(match.group(1))
        money_parsed.append({"value": val, "currency": "INR"})
        money_values.add(str(int(val)) if val == int(val) else str(val))
        # Add to list format for nlu_router (use float for normalized amounts)
        entities_list.append({"label": "amount", "value": float(val)})
    
    # Pattern 2: ₹ or Rs followed by number
    pattern2_matches = re.finditer(r"(?:₹|rs\.?|inr|INR)\s*(\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    for match in pattern2_matches:
        val = float(match.group(1))
        money_parsed.append({"value": val, "currency": "INR"})
        money_values.add(str(int(val)) if val == int(val) else str(val))
        entities_list.append({"label": "amount", "value": float(val)})
    
    # Pattern 3: USD format
    pattern3_matches = re.finditer(r"\$\s?(\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    for match in pattern3_matches:
        val = float(match.group(1))
        money_parsed.append({"value": val, "currency": "USD"})
        money_values.add(str(int(val)) if val == int(val) else str(val))
        entities_list.append({"label": "amount", "value": float(val)})
    
    # Pattern 4: Plain number (could be amount in transfer context)
    # Extract plain numbers that aren't part of a longer string and aren't already marked as money
    pattern4_matches = re.finditer(r"\b(\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    plain_numbers_added = set()
    for match in pattern4_matches:
        val = match.group(1)
        # Only add if: not already in money_values, hasn't been added, and not an account number pattern
        if val not in money_values and val not in plain_numbers_added:
            # Don't add standalone 4-6 digit numbers (those are account numbers, not amounts)
            if not (4 <= len(val) <= 6 and "." not in val):
                try:
                    float_val = float(val)
                    money_parsed.append({"value": float_val, "currency": "default"})
                    # store amount as float
                    entities_list.append({"label": "amount", "value": float_val})
                    plain_numbers_added.add(val)
                except ValueError:
                    pass
    
    if money_parsed:
        entities["money_parsed"] = money_parsed
        # Also populate "amount" field with float values for dialogue_handler compatibility
        entities["amount"] = [float(item["value"]) for item in money_parsed]

    # account number: 4- to 6-digit sequences (but exclude those that are money amounts)
    # Look for patterns like "account 1001", "to 1002", "from 1003"
    acct_patterns = [
        r"(?:account|acc|#|from|to)\s*(\d{4,6})",  # Account preceded by keyword
        r"\b\d{4,6}\b",  # Standalone 4-6 digit numbers
    ]
    
    accounts_found = []
    for pattern in acct_patterns:
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            acc_num = match.group(1) if match.lastindex else match.group(0)
            # Only include if it's not a money value we already extracted
            if acc_num not in money_values and acc_num not in accounts_found:
                accounts_found.append(acc_num)
                # Add to list format for nlu_router
                entities_list.append({"label": "account_number", "value": acc_num})
    
    if accounts_found:
        entities["account_number"] = accounts_found

    # Remove duplicates while preserving order
    if "money_parsed" in entities:
        seen = set()
        unique_money = []
        for item in entities["money_parsed"]:
            key = item["value"]
            if key not in seen:
                seen.add(key)
                unique_money.append(item)
        entities["money_parsed"] = unique_money

    # spaCy built-in entities: MONEY (but filter out date false positives)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities.setdefault("spacy_money", []).append(ent.text)
            # Also add to list format
            if {"label": "amount", "value": ent.text} not in entities_list:
                entities_list.append({"label": "amount", "value": ent.text})
    
    # Keep text for fallback analysis
    entities["text"] = text

    # Return list format for nlu_router compatibility
    return entities_list


def extract(text: str):
    """Extract entities and return as dict format for dialogue_handler.
    
    Converts from list format (nlu_router) to dict format (dialogue_handler).
    
    Args:
        text: User input text
        
    Returns:
        dict: Dictionary with format {"account_number": [...], "amount": [...], ...}
    """
    if not text or not isinstance(text, str):
        return {"account_number": [], "amount": [], "money_parsed": [], "text": ""}
    
    entities_list = extract_entities(text)
    
    # Convert list format to dict format
    result = {"text": text}
    if not entities_list or not isinstance(entities_list, list):
        return result
    
    for entity in entities_list:
        if isinstance(entity, dict):
            label = entity.get("label", "").lower()
            value = entity.get("value", "")
            if label and value:
                if label not in result:
                    result[label] = []
                result[label].append(value)
    
    return result


if __name__ == "__main__":
    while True:
        msg = input("User: ")
        if msg.lower() in ("quit", "exit"):
            break
        print(extract_entities(msg))
