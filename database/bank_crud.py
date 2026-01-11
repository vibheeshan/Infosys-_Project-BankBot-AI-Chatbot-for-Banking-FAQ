
# database/bank_crud.py

from database.db import get_conn
from database.security import hash_password, verify_password
from datetime import datetime


def _ensure_transaction_pin_column():
    """Ensure the `transaction_pin_hash` column exists in the accounts table.

    This performs a no-op if the column already exists.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(accounts)")
    cols = [r[1] for r in cur.fetchall()]
    if "transaction_pin_hash" not in cols:
        try:
            cur.execute("ALTER TABLE accounts ADD COLUMN transaction_pin_hash BLOB")
            conn.commit()
        except Exception:
            # If ALTER fails for any reason, ignore - schema may differ across environments
            pass
    conn.close()


def set_transaction_pin(account_number: str, pin: str) -> bool:
    """Set or update the transaction PIN for an account (hashed).

    Returns True on success, False otherwise.
    """
    if not account_number or not pin:
        return False
    _ensure_transaction_pin_column()
    conn = get_conn()
    cur = conn.cursor()
    try:
        hashed = hash_password(pin)
        cur.execute("UPDATE accounts SET transaction_pin_hash=? WHERE account_number=?", (hashed, account_number))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False

def create_account(name, acc_no, acc_type, balance, password):
    conn = get_conn()
    cur = conn.cursor()
from database.db import get_conn
from database.security import hash_password, verify_password
from datetime import datetime


def create_account(*args, **kwargs):
    """Create an account.

    Supports two common call orders for backward compatibility:
    - (name, acc_no, acc_type, balance, password)
    - (acc_no, name, initial_balance, pin, account_type)
    Also accepts keyword arguments: account_number, user_name, account_type, balance, password, email
    Returns True on success, False on failure.
    """
    # Normalize inputs
    account_number = kwargs.get("account_number") or kwargs.get("acc_no")
    user_name = kwargs.get("user_name") or kwargs.get("name")
    account_type = kwargs.get("account_type") or kwargs.get("acc_type")
    balance = kwargs.get("balance")
    password = kwargs.get("password")
    email = kwargs.get("email")

    if not account_number and args:
        # Determine ordering by checking which arg looks like an account number
        if len(args) >= 5:
            if str(args[0]).isdigit():
                account_number = str(args[0])
                user_name = args[1]
                balance = float(args[2])
                password = args[3]
                account_type = args[4]
            elif str(args[1]).isdigit():
                user_name = args[0]
                account_number = str(args[1])
                account_type = args[2]
                balance = float(args[3])
                password = args[4]
            else:
                user_name, account_number, account_type, balance, password = args[:5]

    if not all([account_number, user_name, account_type, balance is not None, password]):
        return False

    conn = get_conn()
    cur = conn.cursor()
    try:
        # Insert user with email
        if email:
            cur.execute("INSERT OR IGNORE INTO users(name, email) VALUES (?, ?)", (user_name, email))
        else:
            cur.execute("INSERT OR IGNORE INTO users(name) VALUES (?)", (user_name,))
        
        # Ensure we have a user_id
        cur.execute("SELECT id FROM users WHERE name=?", (user_name,))
        user_row = cur.fetchone()
        user_id = user_row[0] if user_row else None

        pwd_hash = hash_password(password)
        # If schema contains transaction_pin_hash, include it (set same as password by default)
        try:
            tx_hash = hash_password(password)
            cur.execute(
                """
                INSERT OR IGNORE INTO accounts(account_number, user_id, account_type, balance, pin_hash, transaction_pin_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (account_number, user_id, account_type, float(balance), pwd_hash, tx_hash),
            )
        except Exception:
            # Fallback for older schema
            cur.execute(
                """
                INSERT OR IGNORE INTO accounts(account_number, user_id, account_type, balance, pin_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (account_number, user_id, account_type, float(balance), pwd_hash),
            )

        conn.commit()
        return True
    except Exception:
        # If password_hash column doesn't exist, try with pin_hash
        try:
            pwd_hash = hash_password(password)
            cur.execute(
                """
                INSERT OR IGNORE INTO accounts(account_number, user_id, account_type, balance, pin_hash)
                VALUES (?, (SELECT id FROM users WHERE name=?), ?, ?, ?)
                """,
                (account_number, user_name, account_type, float(balance), pwd_hash),
            )
            conn.commit()
            return True
        except Exception:
            return False
    finally:
        conn.close()


def get_account(acc_no):
    conn = get_conn()
    cur = conn.cursor()
    
    # Try first with user_id and pin_hash (new schema)
    cur.execute(
        """
        SELECT account_number, 
               (SELECT name FROM users WHERE id=accounts.user_id) as user_name,
               account_type, balance, pin_hash as password_hash
        FROM accounts WHERE account_number=?
        """,
        (acc_no,),
    )
    row = cur.fetchone()
    
    if not row:
        # Try old schema without user_id
        cur.execute(
            """
            SELECT account_number, user_name, account_type, balance, password_hash
            FROM accounts WHERE account_number=?
            """,
            (acc_no,),
        )
        row = cur.fetchone()
    
    conn.close()
    if not row:
        return None
    return {
        "account_number": row[0],
        "user_name": row[1],
        "account_type": row[2],
        "balance": float(row[3]) if row[3] is not None else 0.0,
        "password_hash": row[4],
    }


def authenticate(account_number: str, password: str):
    """Verify account credentials. Returns dict with account info on success, else None."""
    acct = get_account(account_number)
    if not acct:
        return None
    try:
        if verify_password(password, acct.get("password_hash")):
            # return minimal public info
            return {
                "account_number": acct["account_number"],
                "user_name": acct["user_name"],
                "account_type": acct["account_type"],
                "balance": acct["balance"],
            }
    except Exception:
        return None
    return None


def list_accounts():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT account_number, user_name FROM accounts")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_accounts():
    conn = get_conn()
    cur = conn.cursor()
    # Join accounts with users to get the user_name
    cur.execute("""
        SELECT 
            a.account_number, 
            u.name as user_name, 
            a.account_type, 
            a.balance
        FROM accounts a
        LEFT JOIN users u ON a.user_id = u.id
    """)
    rows = cur.fetchall()
    conn.close()
    accounts = []
    for r in rows:
        accounts.append(
            {
                "account_number": r[0],
                "user_name": r[1],
                "account_type": r[2],
                "balance": float(r[3]) if r[3] is not None else 0.0,
                "status": "Active",
            }
        )
    return accounts


def transfer_money(from_acc, to_acc, amount, transaction_pin=None):
    """Transfer money from one account to another using a transaction PIN.

    Args:
        from_acc: Source account number
        to_acc: Destination account number
        amount: Amount to transfer (numeric)
        transaction_pin: 4-digit transaction PIN provided by user

    Returns:
        str: Success or error message
    """
    # Basic validation
    if not from_acc or not to_acc:
        return "Invalid account details"
    if from_acc == to_acc:
        return "Cannot transfer to the same account"

    conn = get_conn()
    cur = conn.cursor()
    try:
        # Ensure transaction pin column exists (backwards compatible)
        _ensure_transaction_pin_column()

        cur.execute("SELECT balance, transaction_pin_hash, pin_hash FROM accounts WHERE account_number=?", (from_acc,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return "Invalid sender account"

        balance, tx_pin_hash, login_pin_hash = row

        # Verify transaction PIN
        if tx_pin_hash:
            try:
                if not transaction_pin:
                    conn.close()
                    return "Please enter your transaction PIN"
                if not verify_password(transaction_pin, tx_pin_hash):
                    conn.close()
                    return "❌ Invalid transaction PIN"
            except Exception as e:
                conn.close()
                return f"Transaction PIN verification error: {str(e)}"
        else:
            # Backwards compatibility: if transaction PIN not set, allow using account PIN (legacy)
            if not transaction_pin:
                conn.close()
                return "Please enter your transaction PIN"
            try:
                if login_pin_hash and verify_password(transaction_pin, login_pin_hash):
                    # allowed, but recommend setting a separate transaction PIN
                    pass
                else:
                    conn.close()
                    return "❌ Invalid transaction PIN"
            except Exception as e:
                conn.close()
                return f"Transaction PIN verification error: {str(e)}"

        # Check recipient exists
        cur.execute("SELECT 1 FROM accounts WHERE account_number=?", (to_acc,))
        if not cur.fetchone():
            conn.close()
            return "Recipient account not found"

        if balance < amount:
            conn.close()
            return "Insufficient balance"

        # Perform transfer
        cur.execute("UPDATE accounts SET balance = balance - ? WHERE account_number=?", (amount, from_acc))
        cur.execute("UPDATE accounts SET balance = balance + ? WHERE account_number=?", (amount, to_acc))

        cur.execute(
            """
            INSERT INTO transactions(from_account, to_account, amount, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (from_acc, to_acc, amount, datetime.now().isoformat()),
        )

        conn.commit()
        conn.close()
        return "Transfer Successful"
    except Exception as e:
        conn.close()
        return f"Transfer failed: {str(e)}"


def get_balance(account_number):
    """Return numeric balance for an account, or None if not found."""
    acct = get_account(account_number)
    if not acct:
        return None
    try:
        return float(acct["balance"])
    except Exception:
        return None


def check_balance(account_number):
    return get_balance(account_number)


def get_transaction_history(account_number, limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT tx_id, from_account, to_account, amount, timestamp FROM transactions WHERE from_account=? OR to_account=? ORDER BY tx_id DESC LIMIT ?",
        (account_number, account_number, limit),
    )
    rows = cur.fetchall()
    conn.close()
    txs = []
    for r in rows:
        txs.append(
            {
                "id": r[0],
                "from_account": r[1],
                "to_account": r[2],
                "amount": float(r[3]),
                "timestamp": r[4],
                "description": "Transfer",
            }
        )
    return txs


def block_card(card_identifier: str):
    """Simple placeholder to simulate blocking a card by type or last-4.

    Returns a human-readable message.
    """
    if not card_identifier:
        return "❌ Please specify which card to block (debit/credit or last 4 digits)."
    return f"✅ Card ({card_identifier}) has been blocked. If this is a mistake, contact support."

# ============================================================================
# CARD MANAGEMENT FUNCTIONS
# ============================================================================

def get_cards_by_account(account_number: str):
    """Get all cards for a specific account."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT card_id, account_number, card_type, card_number, status, created_at
            FROM cards WHERE account_number=?
            """,
            (account_number,),
        )
        rows = cur.fetchall()
        cards = []
        for r in rows:
            cards.append({
                "card_id": r[0],
                "account_number": r[1],
                "card_type": r[2],
                "card_number": r[3],
                "status": r[4],
                "created_at": r[5],
            })
        return cards
    except Exception as e:
        print(f"Error getting cards: {e}")
        return []
    finally:
        conn.close()


def add_card(account_number: str, card_type: str, card_number: str = None) -> bool:
    """Add a new card to an account."""
    if not account_number or not card_type:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO cards (account_number, card_type, card_number, status)
            VALUES (?, ?, ?, ?)
            """,
            (account_number, card_type, card_number, "Active"),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding card: {e}")
        return False
    finally:
        conn.close()


def update_card_status(card_id: int, status: str) -> bool:
    """Update card status (Active, Blocked, Cancelled, etc.)."""
    if not card_id or not status:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE cards SET status=? WHERE card_id=?
            """,
            (status, card_id),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating card status: {e}")
        return False
    finally:
        conn.close()


# ============================================================================
# LOAN MANAGEMENT FUNCTIONS
# ============================================================================

def get_loans_by_account(account_number: str):
    """Get all loans for a specific account."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT loan_id, account_number, loan_type, principal_amount, interest_rate,
                   tenure_months, monthly_emi, remaining_amount, status, start_date, end_date, created_at
            FROM loans WHERE account_number=?
            """,
            (account_number,),
        )
        rows = cur.fetchall()
        loans = []
        for r in rows:
            loans.append({
                "loan_id": r[0],
                "account_number": r[1],
                "loan_type": r[2],
                "principal_amount": float(r[3]) if r[3] is not None else 0.0,
                "interest_rate": float(r[4]) if r[4] is not None else 0.0,
                "tenure_months": r[5],
                "monthly_emi": float(r[6]) if r[6] is not None else 0.0,
                "remaining_amount": float(r[7]) if r[7] is not None else 0.0,
                "status": r[8],
                "start_date": r[9],
                "end_date": r[10],
                "created_at": r[11],
            })
        return loans
    except Exception as e:
        print(f"Error getting loans: {e}")
        return []
    finally:
        conn.close()


def add_loan(account_number: str, loan_type: str, principal_amount: float, 
             interest_rate: float = 0.0, tenure_months: int = 12, 
             monthly_emi: float = 0.0, start_date: str = None) -> bool:
    """Add a new loan to an account."""
    if not account_number or not loan_type or principal_amount <= 0:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO loans 
            (account_number, loan_type, principal_amount, interest_rate, tenure_months, monthly_emi, remaining_amount, status, start_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (account_number, loan_type, principal_amount, interest_rate, tenure_months, 
             monthly_emi, principal_amount, "Active", start_date or datetime.now().isoformat()),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding loan: {e}")
        return False
    finally:
        conn.close()


def update_loan_status(loan_id: int, status: str) -> bool:
    """Update loan status (Active, Closed, Defaulted, etc.)."""
    if not loan_id or not status:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE loans SET status=? WHERE loan_id=?
            """,
            (status, loan_id),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating loan status: {e}")
        return False
    finally:
        conn.close()


def update_loan_remaining_amount(loan_id: int, remaining_amount: float) -> bool:
    """Update remaining amount for a loan."""
    if loan_id is None or remaining_amount < 0:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE loans SET remaining_amount=? WHERE loan_id=?
            """,
            (remaining_amount, loan_id),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating loan remaining amount: {e}")
        return False
    finally:
        conn.close()