# database/db.py

import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

DB_PATH = Path(__file__).parent.parent / "bankbot.db"
DB_NAME = str(DB_PATH)

def get_conn():
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema with all required tables."""
    conn = get_conn()
    cur = conn.cursor()

    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Accounts table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_number TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        account_type TEXT DEFAULT 'Savings',
        balance REAL DEFAULT 0.0,
        pin_hash BLOB,
        transaction_pin_hash BLOB,
        status TEXT DEFAULT 'Active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Transactions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_account TEXT NOT NULL,
        to_account TEXT NOT NULL,
        amount REAL NOT NULL,
        description TEXT,
        status TEXT DEFAULT 'Completed',
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(from_account) REFERENCES accounts(account_number),
        FOREIGN KEY(to_account) REFERENCES accounts(account_number)
    )
    """)

    # Cards table (for card blocking functionality)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        card_id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number TEXT NOT NULL,
        card_type TEXT,
        card_number TEXT,
        status TEXT DEFAULT 'Active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(account_number) REFERENCES accounts(account_number)
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_NAME}")

# Sample data for demo
DEMO_ACCOUNTS = {
    "1001": {"name": "Rajesh Kumar", "balance": 50000.0, "pin": "1234", "transaction_pin": "4321", "account_type": "Savings", "password": "1234"},
    "1002": {"name": "Priya Sharma", "balance": 75000.0, "pin": "5678", "transaction_pin": "8765", "account_type": "Current", "password": "5678"},
    "1003": {"name": "Amit Patel", "balance": 30000.0, "pin": "9012", "transaction_pin": "2109", "account_type": "Savings", "password": "9012"},
}

def seed_demo_data():
    """Seed the database with demo accounts and transactions."""
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        # Clear existing data
        cur.execute("DELETE FROM transactions")
        cur.execute("DELETE FROM cards")
        cur.execute("DELETE FROM accounts")
        cur.execute("DELETE FROM users")
        
        # Add users and accounts
        for acc_no, data in DEMO_ACCOUNTS.items():
            # Insert user
            cur.execute("INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)",
                       (data["name"], f"{data['name'].lower().replace(' ', '_')}@bankbot.com"))
            
            # Fetch user id (works whether user was inserted or existed)
            cur.execute("SELECT id FROM users WHERE name=?", (data["name"],))
            user_row = cur.fetchone()
            user_id = user_row[0] if user_row else None

            # Insert account with both pin_hash and transaction_pin_hash
            pin_hash = bcrypt.hashpw(data["pin"].encode(), bcrypt.gensalt())
            tx_pin = data.get("transaction_pin") or data.get("pin")
            tx_hash = bcrypt.hashpw(tx_pin.encode(), bcrypt.gensalt())
            cur.execute("""
                INSERT OR IGNORE INTO accounts 
                (account_number, user_id, account_type, balance, pin_hash, transaction_pin_hash) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (acc_no, user_id, data["account_type"], data["balance"], pin_hash, tx_hash))
            
            # Insert demo cards
            cur.execute("""
                INSERT INTO cards (account_number, card_type, status)
                VALUES (?, ?, ?)
            """, (acc_no, "Debit", "Active"))
            cur.execute("""
                INSERT INTO cards (account_number, card_type, status)
                VALUES (?, ?, ?)
            """, (acc_no, "Credit", "Active"))
        
        # Add some sample transactions
        sample_transactions = [
            ("1001", "1002", 5000, "Payment from Rajesh to Priya"),
            ("1002", "1003", 2000, "Refund from Priya to Amit"),
        ]
        
        for from_acc, to_acc, amount, desc in sample_transactions:
            cur.execute("""
                INSERT INTO transactions (from_account, to_account, amount, description)
                VALUES (?, ?, ?, ?)
            """, (from_acc, to_acc, amount, desc))
        
        conn.commit()
        print("Demo data seeded successfully")
    except Exception as e:
        print(f"Error seeding demo data: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    seed_demo_data()

