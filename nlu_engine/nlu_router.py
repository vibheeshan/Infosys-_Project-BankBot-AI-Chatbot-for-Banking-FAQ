"""
NLU Router for BankBot AI.
Central routing function that processes user queries through the NLU pipeline.
"""

from nlu_engine.infer_intent import predict_intent
from nlu_engine.entity_extractor import extract_entities
from typing import Dict, Any, List
from database import bank_crud

# Optional Groq LLM helper (used for non-banking queries)
try:
    from experiments.llm_groq import get_groq_response
except Exception:
    get_groq_response = None


# Banking intents that MUST be handled by rule-based dialogue manager.
# All other intents or unknowns will be routed to Groq LLM.
BANKING_INTENTS = {
    "check_balance",
    "transfer_money",
    "card_block",
    "find_atm",
    "loan_inquiry",
    "card_activation",
}

# Session management
SESSION_CONTEXT: Dict[str, Dict[str, Any]] = {}

REQUIRED_SLOTS = {
    "check_balance": ["account_number"],
    "transfer_money": ["from_account", "to_account", "amount", "transaction_pin"],
    "card_block": ["account_number"],
    "transaction_history": ["account_number"]
}


def reset_context(session_id: str):
    """Reset session context."""
    SESSION_CONTEXT[session_id] = {
        "intent": None,
        "slots": {},
    }


def get_context(session_id: str):
    """Get or create session context."""
    if session_id not in SESSION_CONTEXT:
        reset_context(session_id)
    return SESSION_CONTEXT[session_id]


def normalize_entities(entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert list of entity dicts to a normalized dict.
    Extract entities: {label -> [values]}
    """
    result = {"text": ""}
    if not entities_list:
        return result
    
    for entity in entities_list:
        if isinstance(entity, dict):
            label = entity.get("label", "").lower().replace(" ", "_")
            value = entity.get("value", "")
            
            # Normalize label names (convert spaces to underscores)
            if label in ["amount", "money"]:
                label = "amount"
            elif label in ["account_number", "accountnumber", "account number"]:
                label = "account_number"
            elif label in ["transaction_id", "transactionid", "transaction id"]:
                label = "transaction_id"
            elif label in ["to_account", "toaccount", "to account"]:
                label = "to_account"
            
            if label and value:
                if label not in result:
                    result[label] = []
                result[label].append(value)
    
    return result


def map_entities(slots: dict, entities: dict, current_intent: str = None) -> None:
    """Map extracted entities to dialogue slots."""
    if not entities:
        return
    
    # Special handling: if we have a low-confidence amount that looks like an account number
    # in the context of banking intents, intelligently assign it to the right slot
    if current_intent in ["check_balance", "transfer_money", "card_block", "transaction_history"]:
        if "amount" in entities and entities["amount"]:
            for amt in entities["amount"]:
                # If it looks like an account number (4-12 digits with no decimals)
                if isinstance(amt, str) and amt.replace(".", "").isdigit():
                    val = amt.replace(".", "")
                    if 4 <= len(val) <= 12 and "." not in amt:
                        # Determine which slot to fill based on what's missing
                        if current_intent == "transfer_money":
                            # If we already have from_account, this must be to_account
                            if "from_account" in slots and "to_account" not in slots:
                                if "to_account" not in entities:
                                    entities["to_account"] = []
                                if amt not in entities["to_account"]:
                                    entities["to_account"].append(amt)
                            # If we don't have from_account yet, this is from_account
                            elif "from_account" not in slots:
                                if "account_number" not in entities:
                                    entities["account_number"] = []
                                if amt not in entities["account_number"]:
                                    entities["account_number"].append(amt)
                        else:
                            # For other intents, treat as account_number
                            if "account_number" not in entities:
                                entities["account_number"] = []
                            if amt not in entities["account_number"]:
                                entities["account_number"].append(amt)
    
    # Map account_number - but handle transfer_money specially
    if "account_number" in entities and entities["account_number"]:
        val = str(entities["account_number"][0])
        
        if current_intent == "transfer_money":
            # For transfer: intelligently assign accounts
            # If we don't have from_account, this is the sender
            if "from_account" not in slots:
                slots["from_account"] = val
            # If we already have from_account but not to_account, this is recipient
            elif "to_account" not in slots:
                slots["to_account"] = val
        else:
            # For other intents, account_number is the main identifier
            if "account_number" not in slots:
                slots["account_number"] = val
            if "from_account" not in slots:
                slots["from_account"] = val

    # Map to_account - explicit handling for transfer recipient
    if "to_account" in entities and entities["to_account"]:
        new_to = str(entities["to_account"][0])
        if "to_account" not in slots:
            slots["to_account"] = new_to
            # New receiver invalidates previous PIN
            slots.pop("transaction_pin", None)
        else:
            # If receiver changed, update and remove PIN
            if str(slots.get("to_account")) != new_to:
                slots["to_account"] = new_to
                slots.pop("transaction_pin", None)
    
    # Also check if we extracted an account number and need to_account
    # This handles cases where user provides account number but context expects to_account
    if (current_intent == "transfer_money" and 
        "from_account" in slots and 
        "to_account" not in slots and 
        "account_number" in entities and 
        entities["account_number"] and
        len(entities["account_number"]) > 1):
        # Use second account number as to_account if we have multiple
        slots["to_account"] = str(entities["account_number"][1])
        # New receiver invalidates previous PIN
        slots.pop("transaction_pin", None)

    # Map amount (real currency amounts, not mistaken account numbers)
    if "amount" in entities and entities["amount"]:
        try:
            # Only use if it contains currency indicator or is clearly a decimal
            amt_str = str(entities["amount"][0])
            amt_val = float(amt_str.replace("INR", "").replace("USD", "").strip())
            if "amount" not in slots:
                slots["amount"] = amt_val
                # New amount invalidates previous PIN
                slots.pop("transaction_pin", None)
            else:
                # If amount changed, update and remove PIN
                if float(slots.get("amount", 0)) != amt_val:
                    slots["amount"] = amt_val
                    slots.pop("transaction_pin", None)
        except (ValueError, TypeError):
            pass


def missing_slot(intent: str, slots: dict) -> str:
    """Check if required slots are missing for an intent."""
    if intent not in REQUIRED_SLOTS:
        return None
    for slot in REQUIRED_SLOTS[intent]:
        if slot not in slots or slots.get(slot) in (None, ""):
            return slot
    return None


def get_next_required_slot_transfer(slots: dict) -> str:
    """
    For transfer_money, return the NEXT required slot in STRICT order.
    Order: from_account → to_account → amount → transaction_pin
    """
    if not slots.get("from_account"):
        return "from_account"
    elif not slots.get("to_account"):
        return "to_account"
    elif not slots.get("amount"):
        return "amount"
    elif not slots.get("transaction_pin"):
        return "transaction_pin"
    return None


def handle_dialogue(session_id: str, intent: str, entities: Dict[str, Any]) -> str:
    """
    Handle dialogue state and generate response.
    """
    ctx = get_context(session_id)
    slots = ctx["slots"]
    
    if entities is None:
        entities = {"text": ""}
    
    user_text = entities.get("text", "").lower()

    # Override intent based on user keywords (only reset if we detect a NEW intent)
    if any(word in user_text for word in ["balance", "check", "check balance", "check_balance"]):
        if ctx["intent"] != "check_balance":  # Only reset if changing intents
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]  # CRITICAL: Re-assign slots after reset
            ctx["intent"] = "check_balance"
    elif any(word in user_text for word in ["transfer", "send", "pay"]):
        if ctx["intent"] != "transfer_money":
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]  # CRITICAL: Re-assign slots after reset
            ctx["intent"] = "transfer_money"
    elif any(word in user_text for word in ["block", "card"]):
        if ctx["intent"] != "card_block":
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]  # CRITICAL: Re-assign slots after reset
            ctx["intent"] = "card_block"
    elif any(word in user_text for word in ["transaction", "history"]):
        if ctx["intent"] != "transaction_history":
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]  # CRITICAL: Re-assign slots after reset
            ctx["intent"] = "transaction_history"

    # Set intent from NLU if not already set
    if ctx["intent"] is None and intent and intent != "unknown":
        ctx["intent"] = intent

    if ctx["intent"] is None:
        return "I didn't understand. You can: Check balance, Transfer money, Block card, View transaction history"

    # Special handling for transfer: if user mentions "to" or "recipient" with a number, 
    # that's likely the recipient account
    if ctx["intent"] == "transfer_money":
        has_account = "account_number" in entities and entities.get("account_number")
        has_to_pattern = " to " in user_text or user_text.endswith(" to") or " to" in user_text
        if has_account and has_to_pattern:
            if "to_account" not in slots and entities.get("account_number"):
                # This account number is for the recipient
                if "to_account" not in entities:
                    entities["to_account"] = []
                entities["to_account"] = entities["account_number"]
                # Clear account_number so map_entities doesn't use it as from_account
                entities["account_number"] = []

    # Fill slots from extracted entities
    # For numeric-only inputs, defer entity filling to the numeric-only block below (to enforce strict slot order)
    # For mixed inputs, fill all entities normally
    is_numeric_only = user_text.strip().replace(".", "").replace(",", "").isdigit()
    if not is_numeric_only:
        map_entities(slots, entities, ctx["intent"])

    # Check for missing required slots
    missing = missing_slot(ctx["intent"], slots)
    
    # For transfer_money, determine NEXT required slot (STRICT ORDER)
    if ctx["intent"] == "transfer_money":
        next_required = get_next_required_slot_transfer(slots)
        missing = next_required  # Override missing with strict order slot
    
    # If user input is numeric-only and a slot is missing, treat it as slot value
    if is_numeric_only and missing and ctx["intent"] == "transfer_money":
        # For transfer_money, use STRICT order enforcement
        next_required = get_next_required_slot_transfer(slots)
        
        if next_required == "from_account":
            slots["from_account"] = user_text.strip()
            missing = get_next_required_slot_transfer(slots)
        elif next_required == "to_account":
            slots["to_account"] = user_text.strip()
            # If we filled to_account, remove stale PIN
            if slots.get("transaction_pin"):
                slots.pop("transaction_pin", None)
            missing = get_next_required_slot_transfer(slots)
        elif next_required == "amount":
            try:
                slots["amount"] = float(user_text.replace(",", ""))
                # New amount invalidates previous PIN
                if slots.get("transaction_pin"):
                    slots.pop("transaction_pin", None)
                missing = get_next_required_slot_transfer(slots)
            except Exception:
                pass
        elif next_required == "transaction_pin":
            # Only accept 4-digit numeric transaction PINs
            if user_text.strip().isdigit() and len(user_text.strip()) == 4:
                slots["transaction_pin"] = user_text.strip()
            missing = get_next_required_slot_transfer(slots)
    
    elif is_numeric_only and missing and ctx["intent"] != "transfer_money":
        if missing == "transaction_pin":
            # Only accept 4-digit numeric transaction PINs
            candidate = user_text.strip()
            if candidate.isdigit() and len(candidate) == 4:
                slots["transaction_pin"] = candidate
            missing = missing_slot(ctx["intent"], slots)
        elif missing in ("to_account", "from_account"):
            slots[missing] = user_text.strip()
            # If we filled to_account, remove stale PIN
            if missing == "to_account" and slots.get("transaction_pin"):
                slots.pop("transaction_pin", None)
            missing = missing_slot(ctx["intent"], slots)
        elif missing == "amount":
            try:
                slots["amount"] = float(user_text.replace(",", ""))
                # New amount invalidates previous PIN
                if slots.get("transaction_pin"):
                    slots.pop("transaction_pin", None)
                missing = missing_slot(ctx["intent"], slots)
            except Exception:
                pass

    # For transfer_money, use STRICT order enforcement
    # For other intents, use standard missing slot check
    if ctx["intent"] == "transfer_money" and missing:
        prompts = {
            "from_account": "🏦 Please provide your account number.",
            "to_account": "👤 Please provide receiver account number.",
            "amount": "💰 How much would you like to transfer?",
            "transaction_pin": "🔐 Please enter your transaction PIN."
        }
        return prompts.get(missing, "Please provide the required information.")
    
    if missing:
        prompts = {
            "account_number": "📊 Please provide your account number.",
            "from_account": "🏦 Please provide your account number.",
            "to_account": "👤 Please provide receiver account number",
            "amount": "💰 How much would you like to transfer?",
            "transaction_pin": "🔐 Please enter your transaction PIN",
            "card_type": "🃏 Which card? (debit/credit)"
        }
        return prompts.get(missing, "Please provide the required information.")

    # ================= HANDLE INTENTS =================
    
    # CHECK BALANCE
    if ctx["intent"] == "check_balance":
        account_num = slots.get("account_number")
        try:
            balance = bank_crud.get_balance(account_num)
            reset_context(session_id)
            if balance is None:
                return "Account not found."
            return f"Your current balance is INR {balance:,.2f}"
        except Exception as e:
            reset_context(session_id)
            return f"Error checking balance: {str(e)}"

    # TRANSFER MONEY
    if ctx["intent"] == "transfer_money":
        try:
            from_acc = slots.get("from_account")
            to_acc = slots.get("to_account")
            amount = slots.get("amount")
            tx_pin = slots.get("transaction_pin")
            
            if not from_acc or not to_acc or not amount or not tx_pin:
                return "Missing transfer details. Please try again."

            msg = bank_crud.transfer_money(from_acc, to_acc, float(amount), tx_pin)
            # Handle bank_crud responses
            if isinstance(msg, str):
                # Invalid PIN -> remove transaction_pin only, keep context and re-ask PIN
                if "Invalid transaction PIN" in msg or msg.startswith("❌ Invalid transaction PIN"):
                    slots.pop("transaction_pin", None)
                    return "❌ Invalid transaction PIN. Please enter your transaction PIN again."
                # Success -> reset context and confirm
                if "Transfer Successful" in msg or "success" in msg.lower():
                    reset_context(session_id)
                    return "✅ Transfer Successful"
                # Insufficient balance -> reset context and inform user
                if "Insufficient" in msg or "insufficient" in msg.lower():
                    reset_context(session_id)
                    return "Insufficient balance"
                # Any other informational message: reset context and return it
                reset_context(session_id)
                return msg
            # Non-string success case
            reset_context(session_id)
            return "✅ Transfer Successful"
        except Exception as e:
            reset_context(session_id)
            return f"Transfer error: {str(e)}"

    # CARD BLOCK
    if ctx["intent"] == "card_block":
        try:
            account_num = slots.get("account_number")
            bank_crud.block_card(account_num)
            reset_context(session_id)
            return "Your card has been blocked successfully."
        except Exception as e:
            reset_context(session_id)
            return f"Error blocking card: {str(e)}"

    # CARD ACTIVATION
    if ctx["intent"] == "card_activation":
        try:
            account_num = slots.get("account_number")
            reset_context(session_id)
            
            if account_num:
                return f"✅ Card activation initiated for account {account_num}. Your card will be activated within 24 hours. You'll receive an SMS confirmation shortly."
            else:
                return "✅ Card activation initiated. Please follow the OTP sent to your registered phone number. Your card will be activated shortly after verification."
        except Exception as e:
            reset_context(session_id)
            return f"Error activating card: {str(e)}"

    # TRANSACTION HISTORY
    if ctx["intent"] == "transaction_history":
        try:
            account_num = slots.get("account_number")
            txs = bank_crud.get_transaction_history(account_num) or []
            reset_context(session_id)
            if not txs:
                return "No transactions found."
            lines = ["Recent Transactions:"]
            for tx in txs:
                direction = "Sent" if tx.get("from_account") == account_num else "Received"
                other = tx.get("to_account") if direction == "Sent" else tx.get("from_account")
                lines.append(f"{direction}: INR {tx.get('amount'):,.2f} with {other}")
            return "\n".join(lines)
        except Exception as e:
            reset_context(session_id)
            return f"Error fetching history: {str(e)}"

    reset_context(session_id)
    
    # Provide helpful fallback for unsupported queries
    user_text_lower = user_text.lower()
    if any(word in user_text_lower for word in ["atm", "branch", "nearest", "location", "address", "office"]):
        return "I can help with banking transactions. For ATM and branch information, please visit our website or call customer support."
    elif any(word in user_text_lower for word in ["loan", "credit", "apply", "interest", "rate"]):
        return "I can help with account and card services. For loan inquiries, please contact our loan department."
    elif any(word in user_text_lower for word in ["help", "support", "question", "faq", "how"]):
        return "I can help you with: Check balance, Transfer money, Block card, View transaction history. What would you like to do?"
    
    return "Something went wrong."


def process_query(session_id: str, text: str) -> Dict[str, Any]:
    """
    Process user query through the complete NLU pipeline.
    
    CRITICAL: Routes non-banking intents to Groq LLM immediately.
    Only banking intents are handled by rule-based dialogue manager.
    
    Args:
        session_id: Unique session identifier
        text: User input text
    
    Returns:
        Dictionary with intent, entities, confidence score, and bot response
    """
    try:
        # Step 1: Intent Recognition
        intents = predict_intent(text, top_n=3)
        intent = intents[0][0] if intents else "unknown"
        confidence = intents[0][1] if intents else 0.0
    except Exception as e:
        print(f"Intent prediction error: {e}")
        intent = "unknown"
        confidence = 0.0

    try:
        # Step 2: Entity Extraction
        entities_list = extract_entities(text)
        entities = normalize_entities(entities_list)
        entities["text"] = text
    except Exception as e:
        print(f"Entity extraction error: {e}")
        entities = {"text": text}

    # STEP 3: CRITICAL - Check if intent is non-banking
    # Route to Groq immediately for non-banking intents (unknown, fallback, general knowledge, chitchat)
    if intent not in BANKING_INTENTS:
        if callable(get_groq_response):
            try:
                response = get_groq_response(text)
            except Exception as e:
                print(f"Groq LLM error: {e}")
                response = "Sorry, I couldn't process that right now. Please try again."
        else:
            response = "I can help with banking queries. Please ask about checking balance, transfers, card blocking, or ATM locations."
    else:
        # STEP 4: Only handle banking intents with rule-based dialogue manager
        try:
            response = handle_dialogue(session_id, intent, entities)
        except Exception as e:
            print(f"Dialogue handling error: {e}")
            import traceback
            traceback.print_exc()
            response = f"❌ Error: {str(e)}"

    return {
        "session_id": session_id,
        "query": text,
        "intent": intent,
        "confidence": confidence,
        "entities": entities_list,
        "response": response
    }
