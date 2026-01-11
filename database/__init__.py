"""
Database module for BankBot AI.
Handles SQLite operations and CRUD functions for banking data.
"""

from database.db import init_db, seed_demo_data, get_conn
import database.bank_crud as bank_crud

__all__ = [
    "init_db",
    "seed_demo_data",
    "get_conn",
    "bank_crud",
]
