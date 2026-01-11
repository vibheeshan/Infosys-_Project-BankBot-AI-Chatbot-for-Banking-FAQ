# database/user_profile.py
"""User profile management functions."""

from database.db import get_conn


def update_user_profile(account_number: str, user_name: str = None, email: str = None) -> bool:
    """Update user profile information (name and/or email)."""
    if not account_number:
        return False
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Get current user_id for this account
        cur.execute("SELECT user_id FROM accounts WHERE account_number=?", (account_number,))
        row = cur.fetchone()
        if not row:
            return False
        
        user_id = row[0]
        
        # Update user info
        if user_name and email:
            cur.execute("UPDATE users SET name=?, email=? WHERE id=?", (user_name, email, user_id))
        elif user_name:
            cur.execute("UPDATE users SET name=? WHERE id=?", (user_name, user_id))
        elif email:
            cur.execute("UPDATE users SET email=? WHERE id=?", (email, user_id))
        else:
            return False
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating user profile: {e}")
        return False
    finally:
        conn.close()


def get_user_email(account_number: str) -> str:
    """Get email for a user associated with an account."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT u.email FROM users u
            JOIN accounts a ON u.id = a.user_id
            WHERE a.account_number = ?
        """, (account_number,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else ""
    except Exception as e:
        print(f"Error getting user email: {e}")
        conn.close()
        return ""


def get_user_profile(account_number: str) -> dict:
    """Get complete user profile information."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT u.id, u.name, u.email, a.account_number, a.account_type, a.balance
            FROM users u
            JOIN accounts a ON u.id = a.user_id
            WHERE a.account_number = ?
        """, (account_number,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "name": row[1],
                "email": row[2],
                "account_number": row[3],
                "account_type": row[4],
                "balance": row[5]
            }
        return {}
    except Exception as e:
        print(f"Error getting user profile: {e}")
        conn.close()
        return {}
