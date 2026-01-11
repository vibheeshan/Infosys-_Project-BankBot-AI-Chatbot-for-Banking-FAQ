"""Dialogue manager using simple NLU (intent classifier + entity extractor).

This module implements session-based slot filling and safe handling for Milestone-2.
It is intentionally dependency-free (rule-based) and modular so unit tests can call
`handle_message(session_id, text)` directly.

Milestone-3: Routes non-banking queries to Groq LLM via domain gate.
"""


from typing import Dict, Any, List
from nlu_engine import intent_classifier, entity_extractor
from nlu_engine.domain_gate import get_domain, should_reset_context
from experiments.llm_handler import generate_llm_response
from database import bank_crud


# --- Session state ---
SESSION_CONTEXT: Dict[str, Dict[str, Any]] = {}


REQUIRED_SLOTS = {
    "check_balance": ["account_number"],
    "transfer_money": ["from_account", "to_account", "amount", "transaction_pin"],
    "block_card": ["account_number"],
    "loan_info": [],
    "find_atm": [],
    "fallback": [],
}

# Banking intents whitelist
BANKING_INTENTS = {
    "check_balance",
    "transfer_money",
    "block_card",
    "loan_info",
    "find_atm",
}


def reset_context(session_id: str) -> None:
    """Reset session context."""
    SESSION_CONTEXT[session_id] = {"intent": None, "slots": {}, "history": []}


def get_context(session_id: str) -> Dict[str, Any]:
    """Get or create session context."""
    if session_id not in SESSION_CONTEXT:
        reset_context(session_id)
    return SESSION_CONTEXT[session_id]


def _fill_slots_from_entities(slots: Dict[str, Any], entities: Dict[str, List[str]], intent: str = None) -> None:
    """Fill slots from extracted entities, adapting to current intent."""
    accs = entities.get("account_number") or []

    if accs:
        # For check_balance, block_card, loan_info: use first account as 'account_number'
        if intent in ("check_balance", "block_card", "loan_info"):
            if not slots.get("account_number"):
                slots["account_number"] = accs[0]

        elif intent == "transfer_money":
            # For transfer_money: assign accounts in strict order.
            # If two accounts present, treat first as from_account and second as to_account
            if len(accs) >= 2:
                if not slots.get("from_account"):
                    slots["from_account"] = accs[0]
                # If to_account is set/changed, remove any stale transaction_pin
                if not slots.get("to_account"):
                    slots["to_account"] = accs[1]
                    slots.pop("transaction_pin", None)
            
            else:
                # Single account: ONLY fill from_account, NEVER fill to_account with same value
                # This ensures strict order: from_account → to_account → amount → pin
                if not slots.get("from_account"):
                    slots["from_account"] = accs[0]
                # Do NOT fill to_account from single account number
                # User must explicitly provide receiver in next prompt

    # amounts
    amts = entities.get("amount") or []
    if amts:
        # If amount provided and it's new/changed, set and remove stale PIN
        try:
            amt_val = float(amts[0])
            # Only accept positive amounts
            if amt_val > 0:
                if not slots.get("amount"):
                    slots["amount"] = amt_val
                    # newly added amount invalidates previous PIN
                    slots.pop("transaction_pin", None)
                else:
                    # changed amount -> update and remove PIN
                    if float(slots.get("amount")) != amt_val:
                        slots["amount"] = amt_val
                        slots.pop("transaction_pin", None)
        except (ValueError, TypeError):
            pass

    # transaction_pin
    txs = entities.get("transaction_pin") or []
    if txs and not slots.get("transaction_pin"):
        # For transfer flows only accept PIN if previous core slots are present
        if intent == "transfer_money":
            if slots.get("from_account") and slots.get("to_account") and slots.get("amount"):
                slots["transaction_pin"] = txs[0]
        else:
            slots["transaction_pin"] = txs[0]

    # password
    pwds = entities.get("password") or []
    if pwds and not slots.get("password"):
        slots["password"] = pwds[0]


def _missing_slot_for_intent(intent: str, slots: Dict[str, Any]) -> Any:
    """Check for missing required slots in STRICT ORDER."""
    req = REQUIRED_SLOTS.get(intent, [])
    for s in req:
        if s not in slots or slots.get(s) in (None, ""):
            return s
    return None


def _get_next_required_slot_transfer(slots: Dict[str, Any]) -> str:
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


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _is_valid_account_number(account: Any) -> bool:
    """Validate account number format (4-6 digits)."""
    if not account:
        return False
    acc_str = str(account).strip()
    return acc_str.isdigit() and 4 <= len(acc_str) <= 6


def _is_valid_amount(amount: Any) -> bool:
    """Validate amount (positive number)."""
    if amount is None:
        return False
    try:
        amt = float(amount)
        return amt > 0
    except (ValueError, TypeError):
        return False


def _is_valid_pin(pin: Any) -> bool:
    """Validate transaction PIN (4 digits)."""
    if not pin:
        return False
    pin_str = str(pin).strip()
    return pin_str.isdigit() and len(pin_str) == 4


def handle_message(session_id: str, text: str, selected_account: str = None) -> str:
    """
    Main entry point for dialogue handling.
    
    Routes banking queries to NLU + dialogue manager.
    Routes non-banking queries to Groq LLM.
    
    Args:
        session_id: Unique session identifier
        text: Raw user input text
        selected_account: Current account context (used as sender for transfers)
        
    Returns:
        Bot response string
    """
    ctx = get_context(session_id)
    slots = ctx["slots"]

    try:
        # Normalize text
        text = text or ""
        text = text.strip()
        
        if not text:
            return "Please enter your message."
        
        # ==================== DOMAIN GATE ====================
        # Determine if this is a banking or non-banking query
        domain = get_domain(text)
        
        # If switching from banking to non-banking, reset context
        if should_reset_context(domain, ctx.get("intent")):
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]
        
        # ROUTE 1: Non-banking queries → Groq LLM
        if domain == "non_banking":
            return generate_llm_response(text)
        
        # ROUTE 2: Banking queries → NLU + Dialogue Manager
        # ==================== NLU PIPELINE ====================
        intent_result = intent_classifier.classify(text)
        if not isinstance(intent_result, tuple) or len(intent_result) < 2:
            intent, conf = "fallback", 0.0
        else:
            intent, conf = intent_result
        
        entities_result = entity_extractor.extract(text)
        if not isinstance(entities_result, dict):
            entities = {}
        else:
            entities = entities_result

        # Check if this is a numeric-only input
        from nlu_engine.domain_gate import is_numeric_only
        is_num_only = is_numeric_only(text)
        
        # Only switch intent if explicit keywords + high confidence
        if not is_num_only and intent and ctx.get("intent") and intent != ctx.get("intent") and conf > 0.8:
            reset_context(session_id)
            ctx = get_context(session_id)
            slots = ctx["slots"]

        # Set intent when starting (only if not numeric-only)
        if not ctx.get("intent") and not is_num_only:
            ctx["intent"] = intent
        elif not ctx.get("intent"):
            return "🤔 I need to know what you'd like to do. Options: Check balance, Transfer money, Block card, or Ask about loans."

        # Inject selected_account ONLY for non-transfer intents
        # For transfer_money, enforce strict order asking for accounts explicitly
        if selected_account:
            if ctx.get("intent") in ("check_balance", "block_card") and not slots.get("account_number"):
                slots["account_number"] = selected_account
            # DO NOT auto-fill from_account for transfers - must follow strict order

        # Fill slots from entities
        # For numeric-only inputs, defer entity filling to the numeric-only block below
        # For mixed inputs, fill all entities normally
        if not is_num_only:
            _fill_slots_from_entities(slots, entities, ctx.get("intent"))

            # Handle explicit "to X" patterns in transfer
            if ctx.get("intent") == "transfer_money" and " to " in text and entities.get("account_number"):
                accs = entities.get("account_number")
                if accs and len(accs) > 0:
                    new_to = accs[-1]
                    if not slots.get("to_account"):
                        slots["to_account"] = new_to
                        slots.pop("transaction_pin", None)
                    else:
                        if str(slots.get("to_account")) != str(new_to):
                            slots["to_account"] = new_to
                            slots.pop("transaction_pin", None)
            
            # Check for same-account transfer immediately after filling
            if ctx.get("intent") == "transfer_money":
                from_acc = slots.get("from_account")
                to_acc = slots.get("to_account")
                if from_acc and to_acc and str(from_acc) == str(to_acc):
                    # Prevent same-account transfers
                    slots.pop("to_account", None)
                    slots.pop("transaction_pin", None)
                    return "👤 Please provide a different receiver account number"

        # Check missing slots
        missing = _missing_slot_for_intent(ctx.get("intent"), slots)
        
        # If input is numeric-only and a slot is missing, treat it as slot value
        if is_num_only and missing and ctx.get("intent") == "transfer_money":
            # For transfer, use STRICT order enforcement
            next_required = _get_next_required_slot_transfer(slots)
            
            if next_required == "from_account":
                # Fill sender account with validation
                if _is_valid_account_number(text.strip()):
                    slots["from_account"] = text.strip()
                    missing = _get_next_required_slot_transfer(slots)
                else:
                    return "❌ Invalid account number format. Please provide a 4-6 digit account number."
            
            elif next_required == "to_account":
                # Fill receiver account with validation and remove stale PIN
                if _is_valid_account_number(text.strip()):
                    # Check for same-account transfer
                    if str(slots.get("from_account")) == str(text.strip()):
                        return "👤 Please provide a different receiver account number (not the same as sender)"
                    slots["to_account"] = text.strip()
                    slots.pop("transaction_pin", None)
                    missing = _get_next_required_slot_transfer(slots)
                else:
                    return "❌ Invalid account number format. Please provide a 4-6 digit account number."
            
            elif next_required == "amount":
                # Fill amount with validation and remove stale PIN
                try:
                    amount_val = float(text.replace(",", ""))
                    if _is_valid_amount(amount_val):
                        slots["amount"] = amount_val
                        slots.pop("transaction_pin", None)
                        missing = _get_next_required_slot_transfer(slots)
                    else:
                        return "❌ Invalid amount. Please enter a positive number."
                except (ValueError, TypeError):
                    return "❌ Invalid amount format. Please enter a valid number."
            
            elif next_required == "transaction_pin":
                # Accept PIN ONLY as 4-digit numeric with validation
                if _is_valid_pin(text.strip()):
                    slots["transaction_pin"] = text.strip()
                    missing = _get_next_required_slot_transfer(slots)
                else:
                    return "❌ Invalid PIN format. Please enter a 4-digit PIN."
            # If any required slot is still missing, report it
            
        elif is_num_only and missing and ctx.get("intent") != "transfer_money":
            # For other intents, use standard logic with validation
            if missing in ("from_account", "to_account", "account_number"):
                if _is_valid_account_number(text.strip()):
                    slots[missing] = text.strip()
                    missing = _missing_slot_for_intent(ctx.get("intent"), slots)
                else:
                    return "❌ Invalid account number format. Please provide a 4-6 digit account number."
            elif missing == "amount":
                try:
                    amount_val = float(text.replace(",", ""))
                    if _is_valid_amount(amount_val):
                        slots["amount"] = amount_val
                        missing = _missing_slot_for_intent(ctx.get("intent"), slots)
                    else:
                        return "❌ Invalid amount. Please enter a positive number."
                except (ValueError, TypeError):
                    return "❌ Invalid amount format. Please enter a valid number."
            elif missing == "transaction_pin":
                if _is_valid_pin(text.strip()):
                    slots["transaction_pin"] = text.strip()
                    missing = _missing_slot_for_intent(ctx.get("intent"), slots)
                else:
                    return "❌ Invalid PIN format. Please enter a 4-digit PIN."

        if missing:
            prompts = {
                "account_number": "Please provide your account number",
                "from_account": "🏦 Please provide your account number",
                "to_account": "👤 Please provide receiver account number",
                "amount": "💰 How much would you like to transfer?",
                "transaction_pin": "🔐 Please enter your transaction PIN",
                "password": "Please enter your password",
            }
            # For transfer_money, use strict order function for the prompt
            if ctx.get("intent") == "transfer_money":
                next_required = _get_next_required_slot_transfer(slots)
                return prompts.get(next_required, "Please provide the required information")
            else:
                return prompts.get(missing, "Please provide the required information")

        # ==================== EXECUTE BANKING INTENTS ====================

        if ctx.get("intent") == "check_balance":
            acct = slots.get("account_number") or slots.get("from_account")
            if not acct or not _is_valid_account_number(acct):
                reset_context(session_id)
                return "❌ Invalid account number. Please provide a valid 4-6 digit account number."
            bal = bank_crud.get_balance(acct)
            reset_context(session_id)
            if bal is None:
                return "❌ Account not found. Please verify your account number."
            return f"📊 Your current balance is ₹{bal:,.2f}"

        if ctx.get("intent") == "transfer_money":
            from_acc = slots.get("from_account")
            to_acc = slots.get("to_account")
            amount = slots.get("amount")
            tx_pin = slots.get("transaction_pin")

            # All slots should be filled at this point, proceed with transfer
            try:
                msg = bank_crud.transfer_money(from_acc, to_acc, float(amount), tx_pin)
            except Exception as e:
                # On unexpected error, reset context and surface error
                reset_context(session_id)
                return f"Transfer error: {str(e)}"

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

        if ctx.get("intent") == "block_card":
            acct = slots.get("account_number") or slots.get("from_account")
            if not acct or not _is_valid_account_number(acct):
                reset_context(session_id)
                return "❌ Invalid account number. Please provide a valid 4-6 digit account number."
            msg = bank_crud.block_card(acct)
            reset_context(session_id)
            return msg

        if ctx.get("intent") == "loan_info":
            # Example static response for Milestone-2
            reset_context(session_id)
            return (
                "💼 Loan products:\n- Home Loan: competitive rates.\n- Personal Loan: quick approval."
            )

        if ctx.get("intent") == "find_atm":
            # Provide a friendly nearby ATM list and customer care contacts.
            reset_context(session_id)
            return (
                "🏧 Nearby ATMs:\n"
                "• SBI ATM – 0.5 km\n"
                "• HDFC ATM – 0.8 km\n"
                "• ICICI ATM – 1.2 km\n\n"
                "📞 Customer Care:\n"
                "📱 1800-123-4567\n"
                "☎️ 044-2345-6789\n\n"
                "Our support team is available 24/7."
            )

        # Fallback for unknown banking intent
        reset_context(session_id)
        return (
            "🤔 I didn't understand. You can:\n- Check balance\n- Transfer money\n- Block card\n- Ask about loan info"
        )

    except Exception as e:
        # Safe error handling
        import traceback
        reset_context(session_id)
        error_msg = str(e)
        traceback.print_exc()
        return f"❌ An error occurred: {error_msg}"

