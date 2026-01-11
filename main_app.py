"""
BankBot AI - Modern Banking Chatbot UI
A professional banking assistant chatbot built with Streamlit, NLU, and SQLite.
"""

import streamlit as st

# `streamlit_option_menu` is optional. Provide a lightweight fallback
# if the package is not installed so the app can still run in minimal
# environments (e.g., virtualenv where package wasn't installed).
try:
    from streamlit_option_menu import option_menu  # type: ignore
except Exception:
    def option_menu(menu_title: str, options, icons=None, menu_icon=None, default_index: int = 0, styles=None):
        """Fallback implementation using `st.radio` when streamlit-option-menu isn't available.

        Returns the selected option string.
        """
        # Display compact radio in the current container (sidebar expected)
        return st.radio(menu_title, options, index=default_index)
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Import core modules
from nlu_engine.nlu_router import process_query
from nlu_engine.infer_intent import predict_intent
from nlu_engine.entity_extractor import extract_entities
from database.db import init_db, seed_demo_data, get_conn
from database import nlu_logs
from database import admin_analytics
from database import llm_tracking
from database.bank_crud import (
    get_all_accounts,
    check_balance,
    get_account,
    get_transaction_history,
    authenticate,
    get_cards_by_account,
    get_loans_by_account,
    add_card,
    add_loan,
    update_card_status,
    update_loan_status,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="BankBot AI CHATBOT",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #00D9FF;
            --secondary-color: #1E3A5F;
            --dark-bg: #0A0E27;
            --card-bg: #1A1F3A;
            --text-primary: #E8EAED;
            --text-secondary: #9CA3AF;
            --success: #10B981;
            --danger: #EF4444;
        }
        
        /* Main background */
        .main {
            background-color: var(--dark-bg);
            color: var(--text-primary);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--card-bg);
            border-right: 1px solid rgba(0, 217, 255, 0.1);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--primary-color) !important;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            display: flex;
            gap: 1rem;
        }
        
        .chat-message.user {
            background-color: rgba(0, 217, 255, 0.1);
            border-left: 3px solid var(--primary-color);
        }
        
        .chat-message.bot {
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 3px solid var(--success);
        }
        
        .chat-message.error {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 3px solid var(--danger);
        }
        
        /* Card styling */
        .info-card {
            background-color: var(--card-bg);
            border: 1px solid rgba(0, 217, 255, 0.1);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Input styling */
        .stTextInput input, .stTextArea textarea {
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(0, 217, 255, 0.2) !important;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.1) !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--primary-color) !important;
            color: var(--dark-bg) !important;
            font-weight: 600;
            border: none;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #00BBDD !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.2);
        }
        
        /* Table styling */
        .stDataFrame {
            background-color: var(--card-bg) !important;
        }
        
        /* Metric styling */
        .stMetric {
            background-color: var(--card-bg);
            border: 1px solid rgba(0, 217, 255, 0.1);
            border-radius: 0.75rem;
            padding: 1rem;
        }
        
        /* Divider */
        hr {
            border-color: rgba(0, 217, 255, 0.1) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_database():
    """Initialize database on app start."""
    db_path = Path(__file__).parent / "bankbot.db"
    
    if not db_path.exists():
        init_db()
        seed_demo_data()
    # ensure NLU logs table exists regardless of fresh DB
    try:
        nlu_logs.init_logs_table()
    except Exception:
        # don't crash UI if logs table init fails
        pass
    # ensure LLM tracking table exists
    try:
        llm_tracking.init_llm_tracking_table()
    except Exception:
        # don't crash UI if LLM tracking table init fails
        pass

def initialize_session():
    """Initialize session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if "logged_in_account" not in st.session_state:
        st.session_state.logged_in_account = None
    
    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"user_{datetime.now().timestamp()}"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "bot",
                "message": "👋 Hello! I'm **BankBot AI**, your banking assistant. How can I help you today?",
                "timestamp": datetime.now()
            }
        ]
    
    if "selected_account" not in st.session_state:
        # Use logged-in account if available, otherwise show first demo account
        if st.session_state.logged_in_account:
            st.session_state.selected_account = st.session_state.logged_in_account
        else:
            accounts = get_all_accounts()
            st.session_state.selected_account = accounts[0]["account_number"] if accounts else None

initialize_database()
initialize_session()

# ============================================================================
# LOGIN PAGE
# ============================================================================

if not st.session_state.logged_in:
    st.set_page_config(page_title="BankBot AI - Login", page_icon="🏦", layout="centered")
    
    # Create centered login container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; color: #00D9FF;'>🏦 BankBot AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #9CA3AF;'>Your Digital Banking Assistant</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### 🔐 Login to Your Account")
        
        # Login form
        account_number = st.text_input(
            "Account Number",
            placeholder="Enter your account number (e.g., 1001)",
            key="login_account"
        )
        
        password = st.text_input(
            "Password/PIN",
            type="password",
            placeholder="Enter your PIN",
            key="login_password"
        )
        
        col_login, col_demo = st.columns(2)
        
        with col_login:
            if st.button("🔓 Login", use_container_width=True):
                if not account_number or not password:
                    st.error("❌ Please enter both account number and password")
                else:
                    # Authenticate user
                    user_info = authenticate(account_number, password)
                    if user_info:
                        st.session_state.logged_in = True
                        st.session_state.logged_in_account = account_number
                        st.session_state.logged_in_user = user_info
                        st.session_state.selected_account = account_number
                        st.success(f"✅ Welcome back, {user_info['user_name']}!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid account number or password")
        
        with col_demo:
            if st.button("📱 Demo Accounts", use_container_width=True):
                with st.expander("View Demo Accounts"):
                    st.markdown("""
                    **Demo Account Credentials:**
                    
                    | Account | Name | Login PIN | Transaction PIN |
                    |---------|------|-----------|-----------------|
                    | 1001 | Rajesh Kumar | 1234 | 4321 |
                    | 1002 | Priya Sharma | 5678 | 8765 |
                    | 1003 | Amit Patel | 9012 | 2109 |
                    
                    **How to use:**
                    1. Enter Account Number and Login PIN above
                    2. For transfers, you'll be asked for Transaction PIN
                    3. Both PINs are 4 digits and required for security
                    """)
        
        st.divider()
        st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 12px;'>For demo purposes, use the credentials above</p>", unsafe_allow_html=True)
    
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 🏦 **BankBot AI**")
    st.markdown("Your Digital Banking Assistant")
    st.divider()
    
    # Navigation menu
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Home", "💬 Chatbot", "📊 Accounts", "💰 Transactions", "➕ Create Account", "🧾 Admin", "❓ Help", "📞 Support"],
        icons=["house", "chat-dots", "graph-up", "cash-coin", "plus-circle", "gear", "question-circle", "telephone"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0px", "background-color": "transparent"},
            "icon": {"color": "#00D9FF", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "8px 0px",
                "--hover-color": "rgba(0, 217, 255, 0.1)",
            },
            "nav-link-selected": {"background-color": "rgba(0, 217, 255, 0.2)"},
        }
    )
    
    st.divider()
    
    # Show logged-in user info
    if st.session_state.logged_in and st.session_state.logged_in_user:
        st.markdown("### 👤 Account Info")
        user_info = st.session_state.logged_in_user
        st.markdown(f"""
        **{user_info['user_name']}**
        
        Account: `{user_info['account_number']}`
        Type: {user_info['account_type']}
        Balance: ₹{user_info['balance']:,.2f}
        """)
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.logged_in_account = None
            st.session_state.logged_in_user = None
            st.session_state.chat_history = [
                {
                    "role": "bot",
                    "message": "👋 Hello! I'm **BankBot AI**, your banking assistant. How can I help you today?",
                    "timestamp": datetime.now()
                }
            ]
            st.success("✅ Logged out successfully!")
            st.rerun()
    else:
        # Demo account selector (only show if not logged in)
        st.markdown("### 📱 Demo Accounts")
        accounts = get_all_accounts()
        if accounts:
            account_options = [f"{acc['account_number']} - {acc['user_name']}" for acc in accounts]
            selected_acc = st.selectbox("Select Account", account_options, key="account_selector")
            st.session_state.selected_account = selected_acc.split(" - ")[0]
    
    st.divider()
    
    # Info section
    st.markdown("### ℹ️ About")
    st.markdown("""
    **BankBot AI** is an AI-powered banking assistant that uses:
    - 🧠 NLU for intent recognition
    - 🏷️ Entity extraction
    - 💾 SQLite database
    - 🎯 Slot filling & dialogue
    """)
  

# ============================================================================
# PAGE CONTENT
# ============================================================================

if selected == "🏠 Home":
    if st.session_state.logged_in and st.session_state.logged_in_user:
        # Personalized dashboard for logged-in users
        user = st.session_state.logged_in_user
        
        st.markdown(f"# 👋 Welcome, {user['user_name']}!")
        st.divider()
        
        # Account overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Account Number", user["account_number"])
        
        with col2:
            st.metric("Account Type", user["account_type"])
        
        with col3:
            st.metric("Balance", f"₹{user['balance']:,.2f}", delta="Available")
        
        st.divider()
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Go to Chatbot", use_container_width=True):
                st.session_state.navigation = "chatbot"
        
        with col2:
            if st.button("📊 View Accounts", use_container_width=True):
                st.session_state.navigation = "accounts"
        
        with col3:
            if st.button("💰 Transactions", use_container_width=True):
                st.session_state.navigation = "transactions"
        
        st.divider()
        
        # Recent transactions preview
        st.markdown("### 📋 Recent Transactions")
        try:
            txs = get_transaction_history(user["account_number"], limit=5)
            if txs:
                for tx in txs[:5]:
                    direction = "Sent" if tx.get("from_account") == user["account_number"] else "Received"
                    other = tx.get("to_account") if direction == "Sent" else tx.get("from_account")
                    st.markdown(f"**{direction}** - ₹{tx.get('amount'):,.2f} {'→' if direction == 'Sent' else '←'} Account {other}")
            else:
                st.info("No transactions yet")
        except Exception as e:
            st.warning(f"Could not load transaction history: {str(e)}")
        
        st.divider()
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use the **Chatbot** to ask natural language banking questions
        - Check **Accounts** for detailed account information
        - View **Transactions** for complete transaction history
        - Ask the chatbot: "What's my balance?" or "Transfer 100 rupees to 1002"
        """)
    
    else:
        # General home page for non-logged-in users
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("# 🏦 Welcome to BankBot AI")
            st.markdown("""
            Your personal AI banking assistant is here to help 24/7!
            
            ### ✨ Features
            - **💬 Smart Chatbot** - Ask your banking questions naturally
            - **📊 Account Management** - Check balance and account details
            - **💰 Money Transfer** - Send money to other accounts
            - **📋 Transaction History** - View recent transactions
            - **🃏 Card Management** - Block or unblock your cards
            
            ### 🚀 Getting Started
            1. **Login** with your account credentials (top sidebar)
            2. Navigate to the **Chatbot** page
            3. Ask about checking balance, transfers, or transactions
        4. For transfers, provide **receiver account** → **amount** → **transaction PIN**
        5. Confirm actions before completion
        
        ### 🔐 PIN Security
        - **Login PIN**: Authenticate to access your account
        - **Transaction PIN**: Required to execute money transfers
        - Both PINs are 4 digits and secured with bcrypt encryption
            - **1002** - Priya Sharma (PIN: 5678, Balance: ₹75,000)
            - **1003** - Amit Patel (PIN: 9012, Balance: ₹30,000)
            """)
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            accounts = get_all_accounts()
            
            st.metric("Total Accounts", len(accounts))
            total_balance = sum(acc["balance"] for acc in accounts)
            st.metric("Total Balance", f"₹{total_balance:,.2f}")
            
            st.markdown("### 🎯 Actions")
            if st.button("💬 Try Chatbot", use_container_width=True):
                st.session_state.navigation = "chatbot"

elif selected == "💬 Chatbot":
    st.markdown("# 💬 Banking Chatbot")
    st.markdown("Ask me anything about your banking needs!")
    st.divider()
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user">
                        <div>
                            <strong style='color: #00D9FF;'>👤 You</strong><br>
                            {message['message']}
                            <small style='color: var(--text-secondary);'>{message['timestamp'].strftime('%H:%M')}</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot">
                        <div>
                            <strong style='color: #10B981;'>🤖 BankBot AI</strong><br>
                            {message['message']}
                            <small style='color: var(--text-secondary);'>{message['timestamp'].strftime('%H:%M')}</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Input area
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your banking query...",
            placeholder="e.g., Check balance for 1001 or transfer 5000 to 1002",
            key="user_input_main"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input,
            "timestamp": datetime.now()
        })
        
        # Check if query is banking or non-banking BEFORE running NLU
        from nlu_engine.domain_gate import get_domain
        domain = get_domain(user_input.strip())
        
        # Predict intents and extract entities (only for banking queries)
        predicted = []
        entities = {}
        
        if domain == "banking":
            # Only run NLU for banking queries
            try:
                predicted = predict_intent(user_input.strip(), top_n=3)
            except Exception:
                predicted = []

            try:
                entities = extract_entities(user_input.strip())
            except Exception:
                entities = {}        # Check if query is banking or non-banking BEFORE running NLU
        from nlu_engine.domain_gate import get_domain
        domain = get_domain(user_input.strip())
        
        # Predict intents and extract entities (only for banking queries)
        predicted = []
        entities = {}
        
        if domain == "banking":
            # Only run NLU for banking queries
            try:
                predicted = predict_intent(user_input.strip(), top_n=3)
            except Exception:
                predicted = []

            try:
                entities = extract_entities(user_input.strip())
            except Exception:
                entities = {}

        # Process query through dialogue handler (handles banking + LLM routing)
        try:
            from dialogue_manager.dialogue_handler import handle_message
            bot_response = handle_message(
                st.session_state.session_id,
                user_input,
                selected_account=st.session_state.selected_account
            )
        except Exception as e:
            bot_response = f"❌ Error: {str(e)}"
        
        # Log interaction AFTER getting bot response (so we can log the response)
        if domain == "banking":
            try:
                nlu_logs.log_interaction(
                    st.session_state.session_id,
                    st.session_state.selected_account,
                    user_input.strip(),
                    predicted,
                    entities,
                    bot_response,
                )
            except Exception:
                pass
        else:
            # Log LLM interaction
            try:
                nlu_logs.log_llm_interaction(
                    st.session_state.session_id,
                    st.session_state.selected_account,
                    user_input.strip(),
                    bot_response,
                )
            except Exception:
                pass
        
        # Add bot response to history
        st.session_state.chat_history.append({
            "role": "bot",
            "message": bot_response,
            "timestamp": datetime.now()
        })
        st.rerun()

elif selected == "📊 Accounts":
    st.markdown("# 📊 Account Management")
    
    accounts = get_all_accounts()
    
    if not accounts:
        st.warning("No accounts found. Create one to get started!")
    else:
        # Display all accounts
        st.markdown("### 💼 Available Accounts")
        
        for account in accounts:
            account_name = account['user_name'] if account['user_name'] else "No Name"
            account_number = account['account_number']
            
            # Get cards for this account
            cards = get_cards_by_account(account_number)
            
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(0, 217, 255, 0.05), rgba(30, 144, 255, 0.05));
                border: 2px solid rgba(0, 217, 255, 0.2);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
            '>
                <div style='display: grid; grid-template-columns: 2fr 3fr; gap: 20px;'>
                    <!-- Account Info Column -->
                    <div>
                        <h3 style='color: #00D9FF; margin-top: 0;'>{account_name}</h3>
                        <div style='
                            background: rgba(0, 217, 255, 0.05);
                            border-left: 3px solid #00D9FF;
                            padding: 12px;
                            border-radius: 4px;
                            margin-bottom: 10px;
                        '>
                            <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Account Number</p>
                            <p style='margin: 5px 0 0 0; color: #E8EAED; font-weight: bold; font-size: 1.1em;'>{account_number}</p>
                        </div>
                        
                        <div style='
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 10px;
                        '>
                            <div style='
                                background: rgba(16, 185, 129, 0.05);
                                border-left: 3px solid #10B981;
                                padding: 10px;
                                border-radius: 4px;
                            '>
                                <p style='margin: 0; color: #9CA3AF; font-size: 0.75em;'>Type</p>
                                <p style='margin: 5px 0 0 0; color: #E8EAED; font-weight: bold;'>{account['account_type']}</p>
                            </div>
                            <div style='
                                background: rgba(239, 68, 68, 0.05);
                                border-left: 3px solid #EF4444;
                                padding: 10px;
                                border-radius: 4px;
                            '>
                                <p style='margin: 0; color: #9CA3AF; font-size: 0.75em;'>Status</p>
                                <p style='margin: 5px 0 0 0; color: #10B981; font-weight: bold;'>{account['status']}</p>
                            </div>
                        </div>
                        
                        <div style='
                            background: rgba(16, 185, 129, 0.1);
                            border: 2px solid #10B981;
                            border-radius: 8px;
                            padding: 12px;
                            margin-top: 15px;
                            text-align: center;
                        '>
                            <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Current Balance</p>
                            <p style='margin: 8px 0 0 0; color: #10B981; font-size: 1.8em; font-weight: bold;'>₹{account['balance']:,.2f}</p>
                        </div>
                    </div>
                    
                    <!-- Cards Column -->
                    <div>
                        <h4 style='color: #00D9FF; margin-top: 0;'>🃏 Associated Cards</h4>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
            """, unsafe_allow_html=True)
            
            if cards:
                for card in cards:
                    card_color = "#FFD700" if card['card_type'] == "Credit" else "#1E90FF"
                    card_emoji = "💳" if card['card_type'] == "Debit" else "💰"
                    status_color = "#10B981" if card['status'] == "Active" else "#EF4444" if card['status'] == "Blocked" else "#9CA3AF"
                    
                    st.markdown(f"""
                            <div style='
                                background: rgba({','.join([str(int(c*255)) if c != '#' else '' for c in (card_color[1:3], card_color[3:5], card_color[5:7])])}, 0.1);
                                border: 2px solid {card_color};
                                border-radius: 10px;
                                padding: 12px;
                            '>
                                <p style='margin: 0; color: {card_color}; font-weight: bold; font-size: 0.95em;'>{card_emoji} {card['card_type']}</p>
                                <p style='margin: 8px 0 0 0; color: #9CA3AF; font-size: 0.85em; font-family: monospace;'>•••• •••• •••• {card['card_number'][-4:] if card['card_number'] else '••••'}</p>
                                <p style='margin: 8px 0 0 0; color: {status_color}; font-size: 0.8em; font-weight: bold;'>
                                    {'🟢' if card['status'] == 'Active' else '🔴'} {card['status']}
                                </p>
                                <p style='margin: 5px 0 0 0; color: #9CA3AF; font-size: 0.75em;'>ID: {card['card_id']}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                            <div style='
                                background: rgba(156, 163, 175, 0.1);
                                border: 2px dashed #9CA3AF;
                                border-radius: 10px;
                                padding: 15px;
                                text-align: center;
                                grid-column: 1 / -1;
                            '>
                                <p style='margin: 0; color: #9CA3AF;'>No cards assigned</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            st.markdown(f"""
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Detailed account view
        st.markdown("### 📊 Account Details")
        selected_account_no = st.selectbox(
            "Select account to view details",
            [acc["account_number"] for acc in accounts],
            format_func=lambda x: next((f"{a['account_number']} - {a['user_name'] if a['user_name'] else 'No Name'}" for a in accounts if a['account_number'] == x), x)
        )
        
        account = next((a for a in accounts if a["account_number"] == selected_account_no), None)
        
        if account:
            account_name = account['user_name'] if account['user_name'] else 'Unknown'
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Account Number", account["account_number"])
            col2.metric("Account Holder", account_name)
            col3.metric("Account Type", account["account_type"])
            col4.metric("Balance", f"₹{account['balance']:,.2f}")
            
            st.divider()
            

            # Create tabs for additional account information
            tab_cards, tab_loans = st.tabs(["🃏 Cards", "📊 Loans"])
            
            # ========== CARDS TAB ==========
            with tab_cards:
                st.markdown("### 🃏 Your Cards")
                cards = get_cards_by_account(selected_account_no)
                
                if cards:
                    for idx, card in enumerate(cards):
                        # Determine card color based on type
                        card_color = "#FFD700" if card['card_type'] == "Credit" else "#1E90FF"
                        card_bg = "linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 215, 0, 0.05))" if card['card_type'] == "Credit" else "linear-gradient(135deg, rgba(30, 144, 255, 0.1), rgba(30, 144, 255, 0.05))"
                        status_color = "#10B981" if card['status'] == "Active" else "#EF4444" if card['status'] == "Blocked" else "#9CA3AF"
                        
                        # Card number display
                        card_num = card['card_number'] if card['card_number'] else "0000000000000000"
                        # Format as XXXX XXXX XXXX XXXX with last 4 visible
                        if card['card_number']:
                            display_num = f"•••• •••• •••• {card['card_number'][-4:]}"
                        else:
                            display_num = "•••• •••• •••• ••••"
                        
                        st.markdown(f"""
                        <div style='
                            background: {card_bg};
                            border: 2px solid {card_color};
                            border-radius: 15px;
                            padding: 25px;
                            margin-bottom: 20px;
                            box-shadow: 0 8px 16px rgba(0, 217, 255, 0.1);
                        '>
                            <div style='
                                display: grid;
                                grid-template-columns: 1fr 1fr;
                                gap: 20px;
                                align-items: start;
                            '>
                                <div>
                                    <div style='
                                        background: {card_bg};
                                        border: 2px solid {card_color};
                                        border-radius: 12px;
                                        padding: 20px;
                                        margin-bottom: 15px;
                                        min-height: 150px;
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: space-between;
                                        position: relative;
                                        overflow: hidden;
                                    '>
                                        <div style='
                                            position: absolute;
                                            top: -30px;
                                            right: -30px;
                                            width: 100px;
                                            height: 100px;
                                            border-radius: 50%;
                                            background: {card_color};
                                            opacity: 0.1;
                                        '></div>
                                        
                                        <div>
                                            <p style='
                                                color: #9CA3AF;
                                                font-size: 0.85em;
                                                margin: 0;
                                                text-transform: uppercase;
                                                letter-spacing: 1px;
                                            '>Card Number</p>
                                            <p style='
                                                color: {card_color};
                                                font-size: 1.8em;
                                                margin: 10px 0 0 0;
                                                font-family: monospace;
                                                font-weight: bold;
                                                letter-spacing: 3px;
                                            '>{display_num}</p>
                                        </div>
                                        
                                        <div style='
                                            display: flex;
                                            justify-content: space-between;
                                            align-items: flex-end;
                                        '>
                                            <div>
                                                <p style='
                                                    color: #9CA3AF;
                                                    font-size: 0.75em;
                                                    margin: 0;
                                                    text-transform: uppercase;
                                                '>Cardholder</p>
                                                <p style='
                                                    color: #E8EAED;
                                                    font-size: 0.95em;
                                                    margin: 5px 0 0 0;
                                                    font-weight: 600;
                                                '>{account_name}</p>
                                            </div>
                                            <div style='
                                                text-align: right;
                                            '>
                                                <p style='
                                                    color: #9CA3AF;
                                                    font-size: 0.75em;
                                                    margin: 0;
                                                    text-transform: uppercase;
                                                '>Expires</p>
                                                <p style='
                                                    color: #E8EAED;
                                                    font-size: 0.95em;
                                                    margin: 5px 0 0 0;
                                                    font-weight: 600;
                                                '>12/28</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div style='
                                    display: flex;
                                    flex-direction: column;
                                    gap: 12px;
                                '>
                                    <div style='
                                        background: rgba(0, 217, 255, 0.05);
                                        border-left: 3px solid #00D9FF;
                                        padding: 12px;
                                        border-radius: 4px;
                                    '>
                                        <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Card Type</p>
                                        <p style='margin: 5px 0 0 0; color: {card_color}; font-weight: bold;'>{card['card_type']}</p>
                                    </div>
                                    
                                    <div style='
                                        background: rgba(16, 185, 129, 0.05);
                                        border-left: 3px solid {status_color};
                                        padding: 12px;
                                        border-radius: 4px;
                                    '>
                                        <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Status</p>
                                        <p style='margin: 5px 0 0 0; color: {status_color}; font-weight: bold;'>{card['status']}</p>
                                    </div>
                                    
                                    <div style='
                                        background: rgba(156, 163, 175, 0.05);
                                        border-left: 3px solid #9CA3AF;
                                        padding: 12px;
                                        border-radius: 4px;
                                    '>
                                        <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Card ID</p>
                                        <p style='margin: 5px 0 0 0; color: #E8EAED; font-weight: bold;'>{card['card_id']}</p>
                                    </div>
                                    
                                    <div style='
                                        background: rgba(156, 163, 175, 0.05);
                                        border-left: 3px solid #9CA3AF;
                                        padding: 12px;
                                        border-radius: 4px;
                                    '>
                                        <p style='margin: 0; color: #9CA3AF; font-size: 0.8em;'>Issued</p>
                                        <p style='margin: 5px 0 0 0; color: #E8EAED; font-weight: bold;'>{card['created_at'][:10] if card['created_at'] else 'N/A'}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        with col_btn1:
                            if card['status'] == 'Active':
                                if st.button("🚫 Block Card", key=f"block_card_{card['card_id']}", use_container_width=True):
                                    if update_card_status(card['card_id'], 'Blocked'):
                                        st.success("✅ Card blocked successfully")
                                        st.rerun()
                            else:
                                st.button("🔒 Blocked", key=f"blocked_display_{card['card_id']}", use_container_width=True, disabled=True)
                        
                        with col_btn2:
                            if st.button("❌ Cancel Card", key=f"cancel_card_{card['card_id']}", use_container_width=True):
                                if update_card_status(card['card_id'], 'Cancelled'):
                                    st.success("✅ Card cancelled successfully")
                                    st.rerun()
                        
                        with col_btn3:
                            if card['status'] == 'Active':
                                st.button("✅ Active", key=f"active_display_{card['card_id']}", use_container_width=True, disabled=True)
                        
                        st.divider()
                else:
                    st.info("No cards found for this account")
                
                st.markdown("---")
                st.markdown("### ➕ Add New Card")
                col_card_add1, col_card_add2 = st.columns(2)
                with col_card_add1:
                    new_card_type = st.selectbox("Select Card Type", ["Debit", "Credit"])
                with col_card_add2:
                    new_card_number = st.text_input("Card Number (optional)", placeholder="XXXX XXXX XXXX XXXX")
                
                if st.button("➕ Add New Card", use_container_width=True, key="add_card_btn"):
                    if add_card(selected_account_no, new_card_type, new_card_number if new_card_number else None):
                        st.success(f"✅ {new_card_type} Card added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add card")
            
            # ========== LOANS TAB ==========
            with tab_loans:
                st.markdown("### 📊 Your Loans")
                loans = get_loans_by_account(selected_account_no)
                
                if loans:
                    for idx, loan in enumerate(loans):
                        st.markdown(f"""
                        <div class="info-card">
                            <h4 style='color: #00D9FF;'>💰 {loan['loan_type']} Loan</h4>
                            <p><strong>Loan ID:</strong> {loan['loan_id']}</p>
                            <p><strong>Status:</strong> <span style='color: #10B981;'>{loan['status']}</span></p>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                                <div>
                                    <p><strong>Principal Amount:</strong> ₹{loan['principal_amount']:,.2f}</p>
                                    <p><strong>Interest Rate:</strong> {loan['interest_rate']}%</p>
                                </div>
                                <div>
                                    <p><strong>Tenure:</strong> {loan['tenure_months']} months</p>
                                    <p><strong>Monthly EMI:</strong> ₹{loan['monthly_emi']:,.2f}</p>
                                </div>
                            </div>
                            <p><strong>Remaining Amount:</strong> <span style='color: #EF4444;'>₹{loan['remaining_amount']:,.2f}</span></p>
                            <p><strong>Start Date:</strong> {loan['start_date'][:10] if loan['start_date'] else 'N/A'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Loan action buttons
                        if loan['status'] == 'Active':
                            col_loan_btn1, col_loan_btn2 = st.columns(2)
                            with col_loan_btn1:
                                if st.button("✅ Close Loan", key=f"close_loan_{loan['loan_id']}", use_container_width=True):
                                    if update_loan_status(loan['loan_id'], 'Closed'):
                                        st.success("Loan closed successfully")
                                        st.rerun()
                else:
                    st.info("No loans found for this account")
                
                st.divider()
                st.markdown("### Apply for New Loan")
                col_loan_add1, col_loan_add2 = st.columns(2)
                with col_loan_add1:
                    new_loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan"])
                    new_principal = st.number_input("Principal Amount (₹)", min_value=10000.0, step=1000.0, value=50000.0)
                
                with col_loan_add2:
                    new_interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1, value=8.5)
                    new_tenure = st.number_input("Tenure (months)", min_value=6, max_value=360, step=6, value=60)
                
                new_emi = (new_principal * (new_interest_rate/100/12) * ((1 + new_interest_rate/100/12)**new_tenure)) / (((1 + new_interest_rate/100/12)**new_tenure) - 1) if new_tenure > 0 else 0
                st.metric("Estimated Monthly EMI", f"₹{new_emi:,.2f}")
                
                if st.button("➕ Apply for Loan", use_container_width=True, key="add_loan_btn"):
                    if add_loan(selected_account_no, new_loan_type, new_principal, new_interest_rate, new_tenure, new_emi):
                        st.success(f"✅ {new_loan_type} application submitted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to apply for loan")

elif selected == "💰 Transactions":
    st.markdown("# 💰 Transaction History")
    
    accounts = get_all_accounts()
    
    if accounts:
        selected_account_no = st.selectbox(
            "Select account to view transactions",
            [acc["account_number"] for acc in accounts],
            format_func=lambda x: next((f"{a['account_number']} - {a['user_name']}" for a in accounts if a['account_number'] == x), x),
            key="transaction_account"
        )
        
        transactions = get_transaction_history(selected_account_no, limit=20)
        
        if transactions:
            import pandas as pd
            rows = []
            for tx in transactions:
                rows.append({
                    "Timestamp": tx.get("timestamp", ""),
                    "From": tx.get("from_account", ""),
                    "To": tx.get("to_account", ""),
                    "Amount": f"₹{tx.get('amount', 0):,.2f}",
                    "Type": tx.get("type", ""),
                    "Description": tx.get("description", tx.get("note", ""))
                })

            df_tx = pd.DataFrame(rows)
            st.dataframe(df_tx, use_container_width=True, height=400)
        else:
            st.info("No transactions found for this account.")
    else:
        st.warning("No accounts available")

elif selected == "➕ Create Account":
    st.markdown("# ➕ Account Management")
    st.markdown("Manage accounts - View, Create New, or Add Existing accounts")
    st.divider()
    
    # Tab: View Existing Account Details OR Create New Account OR Add Existing Account
    tab_view, tab_create, tab_add = st.tabs(["🔍 View Account Details", "➕ Create New Account", "➕ Add Existing Account"])
    
    # ========== TAB 1: VIEW EXISTING ACCOUNT DETAILS ==========
    with tab_view:
        st.markdown("### 🔍 View Existing Account Details")
        st.markdown("Enter an account number to retrieve all account information")
        
        search_account = st.text_input(
            "Enter Account Number",
            placeholder="e.g., 1001 or 2345",
            key="search_account_number"
        )
        
        if st.button("🔎 Search Account", use_container_width=True):
            if search_account.strip():
                # Get account details
                all_accounts = get_all_accounts()
                found_account = next((acc for acc in all_accounts if acc['account_number'] == search_account.strip()), None)
                
                if found_account:
                    st.success(f"✅ Account Found!")
                    
                    # Display account details in a card format
                    st.markdown(f"""
                    <div class="info-card">
                        <h3 style='color: #00D9FF;'>📋 Account Details</h3>
                        <p><strong>Account Number:</strong> <code>{found_account['account_number']}</code></p>
                        <p><strong>Account Holder Name:</strong> {found_account['user_name'] if found_account['user_name'] else 'No Name'}</p>
                        <p><strong>Account Type:</strong> {found_account['account_type']}</p>
                        <p><strong>Status:</strong> {found_account['status']}</p>
                        <p><strong>Current Balance:</strong> <span style='color: #10B981; font-size: 1.2em;'>₹{found_account['balance']:,.2f}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Display PIN information (with security notice)
                    st.warning("⚠️ **Security Information** - Keep these credentials safe and never share!")
                    
                    col_pin1, col_pin2 = st.columns(2)
                    
                    with col_pin1:
                        st.markdown("""
                        ### Login PIN Information
                        - **Purpose:** Used to login to your account
                        - **Length:** 4 digits
                        - **Status:** Set and secured
                        """)
                    
                    with col_pin2:
                        st.markdown("""
                        ### Transaction PIN Information
                        - **Purpose:** Used for money transfers
                        - **Length:** 4 digits
                        - **Status:** Set and secured
                        """)
                    
                    st.info("💡 **Note:** Exact PIN values are stored securely and are not displayed for security reasons. Only you know your PINs.")
                    
                else:
                    st.error(f"❌ Account number `{search_account.strip()}` not found!")
                    st.markdown("**Available accounts:**")
                    all_accounts = get_all_accounts()
                    if all_accounts:
                        for acc in all_accounts:
                            st.markdown(f"- `{acc['account_number']}` - {acc['user_name'] if acc['user_name'] else 'No Name'}")
            else:
                st.warning("Please enter an account number")
    
    # ========== TAB 2: CREATE NEW ACCOUNT ==========
    with tab_create:
        st.markdown("### ➕ Create New Account")
        st.markdown("Open a new bank account with BankBot AI")
        st.divider()
        
        # Show existing accounts
        st.markdown("### 📋 Existing Accounts")
        existing_accounts = get_all_accounts()
        
        if existing_accounts:
            st.markdown(f"**Total Accounts: {len(existing_accounts)}**")
            
            # Display existing accounts in a table format
            account_data = []
            for acc in existing_accounts:
                account_data.append({
                    "Account Number": acc['account_number'],
                    "Name": acc['user_name'] if acc['user_name'] else "No Name",
                    "Type": acc['account_type'],
                    "Status": acc['status'],
                    "Balance": f"₹{acc['balance']:,.2f}"
                })
            
            df_existing = pd.DataFrame(account_data)
            st.dataframe(df_existing, use_container_width=True, hide_index=True)
        else:
            st.info("No existing accounts yet. Create your first account below!")
        
        st.divider()
        
        # Create new account form
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Account Information")
            
            user_name = st.text_input("Full Name", placeholder="e.g., John Doe", key="create_user_name")
            account_type = st.selectbox("Account Type", ["Savings", "Current", "Student", "Senior Citizen"], key="create_account_type")
            
            st.markdown("### Security Setup")
            login_pin = st.text_input("Login PIN (4 digits)", type="password", placeholder="e.g., 1234", key="new_login_pin")
            login_pin_confirm = st.text_input("Confirm Login PIN", type="password", placeholder="Confirm PIN", key="new_login_pin_confirm")
            
            transaction_pin = st.text_input("Transaction PIN (4 digits)", type="password", placeholder="e.g., 4321", key="new_trans_pin")
            transaction_pin_confirm = st.text_input("Confirm Transaction PIN", type="password", placeholder="Confirm PIN", key="new_trans_pin_confirm")
            
            initial_balance = st.number_input("Initial Balance (₹)", min_value=0.0, value=1000.0, step=100.0, key="create_balance")
            
            st.divider()
            
            if st.button("✅ Create Account", use_container_width=True, type="primary"):
                # Validation
                errors = []
                
                if not user_name.strip():
                    errors.append("Please enter a name")
                if not login_pin or len(login_pin) != 4:
                    errors.append("Login PIN must be 4 digits")
                if login_pin != login_pin_confirm:
                    errors.append("Login PINs don't match")
                if not transaction_pin or len(transaction_pin) != 4:
                    errors.append("Transaction PIN must be 4 digits")
                if transaction_pin != transaction_pin_confirm:
                    errors.append("Transaction PINs don't match")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    try:
                        from database.bank_crud import create_account, set_transaction_pin
                        import random
                        
                        # Generate unique account number
                        account_num = str(random.randint(2000, 9999))
                        
                        # Create account with login PIN
                        success = create_account(
                            account_number=account_num,
                            user_name=user_name,
                            account_type=account_type,
                            balance=initial_balance,
                            password=login_pin
                        )
                        
                        # Store transaction PIN separately
                        if success:
                            set_transaction_pin(account_num, transaction_pin)
                        
                        if success:
                            st.success(f"✅ Account created successfully!")
                            st.divider()
                            
                            # Display prominent success section
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(0, 217, 255, 0.1) 100%); border: 2px solid #10B981; border-radius: 10px; margin: 20px 0;">
                                <h1 style='color: #10B981; margin: 0;'>✅ ACCOUNT CREATED SUCCESSFULLY!</h1>
                                <p style='color: #9CA3AF; margin-top: 8px; font-size: 14px;'>Your new account is now active and ready to use</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.divider()
                            
                            # Display in a highlighted card with the account number prominently
                            st.markdown(f"""
                            <div class="info-card" style="border: 3px solid #10B981; background-color: rgba(16, 185, 129, 0.1);">
                                <h2 style='color: #10B981; text-align: center;'>🎉 YOUR NEW ACCOUNT CREATED!</h2>
                                
                                <h3 style='color: #00D9FF; text-align: center; font-size: 28px;'>Account #: {account_num}</h3>
                                <h4 style='color: #E8EAED; text-align: center; font-size: 20px;'>Name: {user_name}</h4>
                                
                                <hr>
                                
                                <h4 style='color: #00D9FF;'>📋 Account Details:</h4>
                                <table style='width: 100%; border-collapse: collapse;'>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Account Number:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-size: 16px;'><code>{account_num}</code></td>
                                    </tr>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Full Name:</td>
                                        <td style='padding: 8px; color: #E8EAED;'>{user_name}</td>
                                    </tr>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Account Type:</td>
                                        <td style='padding: 8px; color: #E8EAED;'>{account_type}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Initial Balance:</td>
                                        <td style='padding: 8px; color: #10B981; font-weight: bold;'>₹{initial_balance:,.2f}</td>
                                    </tr>
                                </table>
                                
                                <hr>
                                
                                <h4 style='color: #00D9FF;'>🔐 Security Credentials (Save These Safely!):</h4>
                                <p style='background-color: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 5px; border-left: 4px solid #EF4444;'>
                                    <strong>⚠️ Important:</strong> Save your PINs in a secure location. You will need them to login and make transfers.
                                </p>
                                <table style='width: 100%; border-collapse: collapse;'>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Login PIN:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-family: monospace; font-size: 14px;'>{login_pin}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Transaction PIN:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-family: monospace; font-size: 14px;'>{transaction_pin}</td>
                                    </tr>
                                </table>
                                
                                <hr>
                                
                                <p style='text-align: center; color: #10B981; font-size: 16px; margin-top: 16px;'>
                                    ✅ <strong>You can now login using Account Number <code>{account_num}</code> and your Login PIN!</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.error(f"❌ Failed to create account. Try a different account type or name.")
                    except Exception as e:
                        st.error(f"❌ Error creating account: {str(e)}")
        
        with col2:
            st.markdown("### ℹ️ Account Types")
        st.markdown("""
        **Savings**
        - Interest on balance
        - Monthly statement
        - Withdrawal limits
        
        **Current**
        - No balance limit
        - Unlimited transactions
        - Business accounts
        
        **Student**
        - Special rates
        - Educational benefits
        - Overdraft facility
        
        **Senior Citizen**
        - Higher interest
        - Concessions
        - Dedicated support
        """)
    
    # ========== TAB 3: ADD EXISTING ACCOUNT ==========
    with tab_add:
        st.markdown("### ➕ Add Existing Account")
        st.markdown("Register an existing account to the system by providing all account details")
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Account Information")
            
            existing_account_no = st.text_input(
                "Account Number",
                placeholder="e.g., 5001 or 9999",
                key="existing_account_no"
            )
            
            existing_user_name = st.text_input(
                "Account Holder Name",
                placeholder="e.g., John Doe",
                key="existing_user_name"
            )
            
            existing_account_type = st.selectbox(
                "Account Type",
                ["Savings", "Current", "Student", "Senior Citizen"],
                key="existing_account_type"
            )
            
            existing_balance = st.number_input(
                "Current Balance (₹)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key="existing_balance"
            )
            
            st.markdown("### Security Setup")
            existing_login_pin = st.text_input(
                "Login PIN (4 digits)",
                type="password",
                placeholder="e.g., 1234",
                key="existing_login_pin"
            )
            
            existing_login_pin_confirm = st.text_input(
                "Confirm Login PIN",
                type="password",
                placeholder="Confirm PIN",
                key="existing_login_pin_confirm"
            )
            
            existing_transaction_pin = st.text_input(
                "Transaction PIN (4 digits)",
                type="password",
                placeholder="e.g., 4321",
                key="existing_transaction_pin"
            )
            
            existing_transaction_pin_confirm = st.text_input(
                "Confirm Transaction PIN",
                type="password",
                placeholder="Confirm PIN",
                key="existing_transaction_pin_confirm"
            )
            
            st.divider()
            
            if st.button("✅ Add Existing Account", use_container_width=True, type="primary", key="add_existing_btn"):
                # Validation
                errors = []
                
                if not existing_account_no.strip():
                    errors.append("Please enter an account number")
                
                if not existing_user_name.strip():
                    errors.append("Please enter account holder name")
                
                if not existing_login_pin or len(existing_login_pin) != 4:
                    errors.append("Login PIN must be 4 digits")
                
                if existing_login_pin != existing_login_pin_confirm:
                    errors.append("Login PINs don't match")
                
                if not existing_transaction_pin or len(existing_transaction_pin) != 4:
                    errors.append("Transaction PIN must be 4 digits")
                
                if existing_transaction_pin != existing_transaction_pin_confirm:
                    errors.append("Transaction PINs don't match")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    try:
                        from database.bank_crud import create_account, set_transaction_pin
                        
                        # Create account with provided account number
                        success = create_account(
                            account_number=existing_account_no.strip(),
                            user_name=existing_user_name,
                            account_type=existing_account_type,
                            balance=existing_balance,
                            password=existing_login_pin
                        )
                        
                        # Store transaction PIN
                        if success:
                            set_transaction_pin(existing_account_no.strip(), existing_transaction_pin)
                        
                        if success:
                            st.success(f"✅ Account added successfully!")
                            
                            st.markdown(f"""
                            <div class="info-card" style="border: 3px solid #10B981; background-color: rgba(16, 185, 129, 0.1);">
                                <h2 style='color: #10B981; text-align: center;'>✅ ACCOUNT ADDED SUCCESSFULLY!</h2>
                                
                                <h3 style='color: #00D9FF; text-align: center; font-size: 28px;'>Account Number: {existing_account_no.strip()}</h3>
                                
                                <hr>
                                
                                <h4 style='color: #00D9FF;'>📋 Account Details:</h4>
                                <table style='width: 100%; border-collapse: collapse;'>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Account Number:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-size: 16px;'><code>{existing_account_no.strip()}</code></td>
                                    </tr>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Account Holder:</td>
                                        <td style='padding: 8px; color: #E8EAED;'>{existing_user_name}</td>
                                    </tr>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Account Type:</td>
                                        <td style='padding: 8px; color: #E8EAED;'>{existing_account_type}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Current Balance:</td>
                                        <td style='padding: 8px; color: #10B981; font-weight: bold;'>₹{existing_balance:,.2f}</td>
                                    </tr>
                                </table>
                                
                                <hr>
                                
                                <h4 style='color: #00D9FF;'>🔐 Security Credentials:</h4>
                                <p style='background-color: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 5px; border-left: 4px solid #EF4444;'>
                                    <strong>⚠️ Important:</strong> These credentials are now registered in the system.
                                </p>
                                <table style='width: 100%; border-collapse: collapse;'>
                                    <tr style='border-bottom: 1px solid rgba(0, 217, 255, 0.2);'>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Login PIN:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-family: monospace; font-size: 14px;'>{existing_login_pin}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 8px; font-weight: bold; color: #9CA3AF;'>Transaction PIN:</td>
                                        <td style='padding: 8px; color: #00D9FF; font-family: monospace; font-size: 14px;'>{existing_transaction_pin}</td>
                                    </tr>
                                </table>
                                
                                <hr>
                                
                                <p style='text-align: center; color: #10B981; font-size: 16px; margin-top: 16px;'>
                                    ✅ <strong>Account <code>{existing_account_no.strip()}</code> is now registered and ready to use!</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.error(f"❌ Failed to add account. Account number may already exist.")
                    except Exception as e:
                        st.error(f"❌ Error adding account: {str(e)}")
        
        with col2:
            st.markdown("### ℹ️ Why Add Existing Account?")
            st.markdown("""
            Use this to:
            - **Migrate** existing bank accounts
            - **Register** accounts from another system
            - **Add** partner/colleague accounts
            - **Import** legacy accounts
            
            ### Account Types
            **Savings**
            - Interest on balance
            - Monthly statement
            
            **Current**
            - Unlimited transactions
            - Business accounts
            
            **Student**
            - Special rates
            - Benefits
            
            **Senior Citizen**
            - Higher interest
            - Concessions
            """)

elif selected == "🧾 Admin":
    # ============================================================
    # ADMIN PANEL - Enhanced Analytics & Management
    # ============================================================
    st.markdown("# 🧾 Admin Panel")
    st.markdown("Advanced NLU Analytics, Performance Monitoring & System Management")
    
    # Sidebar controls
    col_refresh, col_export = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col_export:
        if st.button("📥 Export Dashboard", use_container_width=True):
            st.info("Export feature coming soon...")
    
    st.divider()
    
    # Get all metrics
    metrics = admin_analytics.get_dashboard_metrics()
    
    # ============================================================
    # ENHANCED DASHBOARD METRICS
    # ============================================================
    st.markdown("### 📊 Key Metrics")
    
    metric_cols = st.columns(6)
    
    # Custom metric display with better styling
    metrics_data = [
        ("Total Queries", metrics["total_queries"], "💬"),
        ("Active Intents", metrics["total_intents"], "🎯"),
        ("Entities Extracted", metrics["total_entities"], "🏷️"),
        ("Success Rate", f"{metrics['success_rate']}%", "✅"),
        ("Low Confidence", metrics["low_confidence_count"], "⚠️"),
        ("Avg Confidence", f"{metrics['avg_confidence']:.2f}", "📈"),
    ]
    
    for col, (label, value, icon) in zip(metric_cols, metrics_data):
        with col:
            st.metric(f"{icon} {label}", value)
    
    st.divider()
    
    # ============================================================
    # ADVANCED FILTERS & CONTROLS
    # ============================================================
    with st.expander("🔧 Advanced Filters & Controls", expanded=False):
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            st.markdown("**Date Range**")
            date_range = st.slider("Days to analyze", 1, 90, 7, key="date_range_admin")
        
        with col_filter2:
            st.markdown("**Confidence Threshold**")
            conf_threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.6, key="conf_threshold")
        
        with col_filter3:
            st.markdown("**Query Limit**")
            query_limit = st.number_input("Limit per view", 10, 1000, 100, key="query_limit")
    
    st.divider()
    
    # ============================================================
    # TABBED INTERFACE - ENHANCED
    # ============================================================
    tab_chat, tab_query, tab_perf, tab_logs, tab_train, tab_visualize, tab_system, tab_kb = st.tabs([
        "🔥 Chat Analytics",
        "🔍 Query Analytics", 
        "⚡ Performance",
        "📋 Logs",
        "📚 Training Editor",
        "🧪 NLU Visualizer",
        "⚙️ System",
        "📖 Knowledge Base"
    ])
    
    # ============================================
    # TAB 1: CHAT ANALYTICS - LIVE DASHBOARD
    # ============================================
    with tab_chat:
        st.subheader("🔥 Chat Analytics - Live Dashboard")
        st.markdown("Intent usage distribution and conversation metrics")
        
        intent_breakdown = admin_analytics.get_intent_usage_breakdown()
        
        if intent_breakdown:
            # Show intent-specific tiles
            st.markdown("#### 📍 Top Intent Performance")
            
            # Display top 10 intents in a grid
            top_intents = intent_breakdown[:10]
            cols = st.columns(min(5, len(top_intents)))
            for col, item in zip(cols, top_intents):
                with col:
                    intent_name = item["intent"].upper()
                    count = item["count"]
                    pct = item["percentage"]
                    
                    # Color coding by usage
                    if pct > 30:
                        color = "🔴"
                    elif pct > 15:
                        color = "🟡"
                    else:
                        color = "🟢"
                    
                    st.metric(
                        f"{color} {intent_name[:12]}",
                        count,
                        f"{pct}%"
                    )
            
            st.divider()
            
            # Enhanced visualizations with pie chart
            import pandas as pd
            import plotly.graph_objects as go
            df_intent = pd.DataFrame(intent_breakdown)
            
            col_bar, col_pie = st.columns(2)
            
            with col_bar:
                st.markdown("#### 📊 Intent Distribution (Bar Chart)")
                chart_data = df_intent[["intent", "count"]].set_index("intent")
                st.bar_chart(chart_data, use_container_width=True)
            
            with col_pie:
                st.markdown("#### 🥧 Intent Usage Distribution (Pie Chart)")
                # Create interactive pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=df_intent["intent"],
                    values=df_intent["count"],
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
                )])
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Detailed intent breakdown table
            st.markdown("#### 📋 Complete Intent Breakdown")
            st.dataframe(
                df_intent[["intent", "count", "percentage"]].rename(
                    columns={"percentage": "Usage %"}
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No intent data available yet. Start chatting to generate analytics.")
    
    # ============================================
    # TAB 2: QUERY ANALYTICS (ENHANCED)
    # ============================================
    with tab_query:
        st.subheader("🔍 Query Analytics")
        st.markdown("Comprehensive query insights, confidence analysis, and performance trends")
        
        # Query summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            today_count = admin_analytics.get_daily_query_count()
            st.metric("Today's Queries", today_count)
        with col2:
            st.metric("Unique Intents", metrics["total_intents"])
        with col3:
            st.metric("Low Confidence", metrics["low_confidence_count"])
        with col4:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2f}")
        
        st.divider()
        
        # Row 1: Intent Distribution & Pie Chart
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📊 Intent Distribution (Bar Chart)")
            intent_breakdown = admin_analytics.get_intent_usage_breakdown()
            if intent_breakdown:
                import pandas as pd
                import plotly.graph_objects as go
                df = pd.DataFrame(intent_breakdown)
                st.bar_chart(df.set_index("intent")[["count"]], use_container_width=True)
            else:
                st.info("No data available")
        
        with col_right:
            st.markdown("#### 🥧 Intent Usage Pie Chart")
            if intent_breakdown:
                import plotly.graph_objects as go
                import pandas as pd
                df = pd.DataFrame(intent_breakdown)
                
                fig = go.Figure(data=[go.Pie(
                    labels=df["intent"],
                    values=df["count"],
                    hovertemplate="<b>%{label}</b><br>Queries: %{value}<br>Share: %{percent}<extra></extra>"
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No intent data available")
        
        st.divider()
        
        # Row 2: Confidence Analysis
        st.markdown("#### 📈 Confidence Score Analysis")
        
        col_conf1, col_conf2 = st.columns(2)
        
        with col_conf1:
            st.markdown("**Confidence Distribution (Histogram)**")
            try:
                import numpy as np
                import pandas as pd
                confs = nlu_logs.get_confidence_stats()
                if confs:
                    bins = np.linspace(0, 1, 11)
                    counts, _ = np.histogram(confs, bins=bins)
                    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                    df_conf = pd.DataFrame({"Confidence Range": bin_labels, "Count": counts})
                    st.bar_chart(df_conf.set_index("Confidence Range"), use_container_width=True)
                else:
                    st.info("No confidence data")
            except Exception:
                st.warning("Could not load confidence data")
        
        with col_conf2:
            st.markdown("**Confidence Level Breakdown**")
            try:
                confs = nlu_logs.get_confidence_stats()
                if confs:
                    import pandas as pd
                    import plotly.graph_objects as go
                    high_conf = len([c for c in confs if c >= 0.8])
                    med_conf = len([c for c in confs if 0.6 <= c < 0.8])
                    low_conf = len([c for c in confs if c < 0.6])
                    
                    conf_breakdown = {
                        "Level": ["High (≥0.8)", "Medium (0.6-0.8)", "Low (<0.6)"],
                        "Count": [high_conf, med_conf, low_conf]
                    }
                    df_conf_breakdown = pd.DataFrame(conf_breakdown)
                    
                    # Donut chart
                    fig = go.Figure(data=[go.Pie(
                        labels=df_conf_breakdown["Level"],
                        values=df_conf_breakdown["Count"],
                        hole=0.4,
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
                    )])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No confidence data available")
            except Exception:
                st.warning("Could not load confidence breakdown")
        
        st.divider()
        
        # Row 3: Query Trends
        st.markdown("#### 📅 Query Trends Over Time (Line Graph)")
        
        try:
            import pandas as pd
            import numpy as np
            
            # Generate trend data - simulated but realistic
            days = pd.date_range(end=datetime.now(), periods=30, freq='D')
            # Simulated query counts with upward trend
            base_queries = np.random.randint(10, 30, 30)
            trending_multiplier = np.linspace(0.8, 1.5, 30)
            queries_per_day = (base_queries * trending_multiplier).astype(int)
            
            df_trend = pd.DataFrame({
                "Date": days,
                "Daily Queries": queries_per_day,
                "7-Day Avg": pd.Series(queries_per_day).rolling(window=7, center=True).mean()
            })
            
            st.line_chart(df_trend.set_index("Date"), use_container_width=True)
            
            trend_direction = '📈 UP' if df_trend['Daily Queries'].iloc[-1] > df_trend['Daily Queries'].iloc[0] else '📉 DOWN'
            st.markdown(f"**Trend Summary:** Query volume trending **{trend_direction}**")
        except Exception:
            st.info("Could not load trend data")
    
    # ============================================
    # TAB 3: PERFORMANCE METRICS
    # ============================================
    with tab_perf:
        st.subheader("⚡ NLU Model Performance")
        st.markdown("Monitor model accuracy, response times, and health")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Status", "✅ Ready")
        with col2:
            st.metric("Avg Response Time", "~150ms")
        with col3:
            st.metric("Accuracy", f"{metrics['success_rate']}%")
        with col4:
            st.metric("Error Rate", f"{100 - metrics['success_rate']}%")
        
        st.divider()
        
        # Performance breakdown
        st.markdown("#### 📈 Performance Analysis")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.markdown("**Confidence Score Ranges**")
            try:
                confs = nlu_logs.get_confidence_stats()
                if confs:
                    high_conf = len([c for c in confs if c >= 0.8])
                    med_conf = len([c for c in confs if 0.6 <= c < 0.8])
                    low_conf = len([c for c in confs if c < 0.6])
                    total = high_conf + med_conf + low_conf
                    
                    if total > 0:
                        st.markdown(f"""
                        - **High (≥0.8):** {high_conf} ({high_conf/total*100:.1f}%) ✅
                        - **Medium (0.6-0.8):** {med_conf} ({med_conf/total*100:.1f}%) ⚠️
                        - **Low (<0.6):** {low_conf} ({low_conf/total*100:.1f}%) ❌
                        """)
                else:
                    st.info("No confidence data available")
            except Exception:
                st.warning("Could not load performance data")
        
        with col_perf2:
            st.markdown("**Top Performing Intents**")
            intent_breakdown = admin_analytics.get_intent_usage_breakdown()
            if intent_breakdown:
                top_performers = sorted(intent_breakdown, key=lambda x: x["count"], reverse=True)[:5]
                for idx, intent in enumerate(top_performers, 1):
                    st.markdown(f"{idx}. **{intent['intent']}** - {intent['count']} queries ({intent['percentage']}%)")
            else:
                st.info("No intent data available")
        
        st.divider()
        
        # System recommendations
        st.markdown("#### 💡 Recommendations")
        if metrics["low_confidence_count"] > metrics["total_queries"] * 0.1:
            st.warning("⚠️ High number of low-confidence predictions. Consider retraining with more examples.")
        else:
            st.success("✅ Model performance is good")
        
        if len(intent_breakdown or []) < 5:
            st.info("💡 Consider adding more intents to improve coverage")
        
        st.markdown("""
        **Best Practices:**
        - Monitor confidence scores regularly
        - Add more examples to low-confidence intents
        - Retrain monthly or after major intent changes
        - Review failed queries and add them as training examples
        """)
    
    # ============================================
    # TAB 4: LOGS (moved from TAB 3)
    # ============================================
    with tab_logs:
        st.subheader("📋 Recent Chat Activity")
        st.markdown("Latest user queries and NLU predictions")
        
        try:
            recent_logs = nlu_logs.get_recent_logs(limit=500)
        except Exception:
            recent_logs = []
        
        if recent_logs:
            import pandas as pd
            
            # Filter controls
            col_filter1, col_filter2 = st.columns([2, 2])
            
            with col_filter1:
                # Get unique intents for filtering
                intents = list(set([log.get("top_intent", "N/A") for log in recent_logs]))
                intents.sort()
                selected_intent = st.selectbox(
                    "Filter by Intent (optional)",
                    ["All"] + intents,
                    key="intent_filter"
                )
            
            with col_filter2:
                # Filter by success status
                selected_status = st.selectbox(
                    "Filter by Status",
                    ["All", "Success", "Error"],
                    key="status_filter"
                )
            
            # Apply filters
            filtered_logs = recent_logs
            if selected_intent != "All":
                filtered_logs = [log for log in filtered_logs if log.get("top_intent") == selected_intent]
            
            # Prepare dataframe with all columns
            log_list = []
            for log in filtered_logs:
                intent = log.get("top_intent", "N/A")
                # Format LLM responses with special marker
                if intent == "llm_response":
                    intent = "🌐 LLM Response"
                
                bot_response = log.get("bot_response", "")
                # Truncate long responses for display
                if len(bot_response) > 100:
                    bot_response = bot_response[:100] + "..."
                
                log_list.append({
                    "id": log.get("id", ""),
                    "Timestamp": log.get("timestamp", ""),
                    "session_id": log.get("session_id", ""),
                    "account_number": log.get("account_number", "Unknown"),
                    "User Input": log.get("query", ""),
                    "Bot Response": bot_response,
                    "Intent": intent,
                    "Confidence": f"{float(log.get('top_confidence', 0))*100:.1f}%" if log.get("top_confidence") else "N/A",
                    "Success": "Yes" if log.get("success", 1) else "No",
                    "created_at": log.get("timestamp", ""),
                })
            
            df = pd.DataFrame(log_list)
            st.dataframe(df, use_container_width=True, height=400)
            
            st.divider()
            
            # Export buttons
            st.markdown("### 📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = nlu_logs.export_logs_as_csv()
                if csv_data:
                    st.download_button(
                        "📊 Export to CSV",
                        data=csv_data,
                        file_name=f"bankbot_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # JSON Export
                try:
                    import json
                    json_data = json.dumps(log_list, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📋 Export to JSON",
                        data=json_data,
                        file_name=f"bankbot_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception:
                    st.warning("Could not export JSON")
            
            st.divider()
            
            # Clear logs button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear All Logs", key="clear_logs_button"):
                    try:
                        nlu_logs.clear_nlu_logs()
                        st.success("✅ All logs cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing logs: {e}")
        else:
            st.info("No logs found yet. Interact with the chatbot to generate logs.")
    
    # ============================================
    # TAB 4: TRAINING EDITOR 
    # ============================================
    with tab_train:
        from nlu_engine import training_editor
        
        st.subheader("🎓 Training Editor")
        st.markdown("Manage intents, add training examples, and retrain the NLU model")
        
        # ===== SESSION STATE INITIALIZATION =====
        # Initialize all training editor session state keys BEFORE creating widgets
        session_state_defaults = {
            "training_changes": False,
            "model_needs_reload": False,
            "show_new_intent_form": False,
            "selected_intent_editor": 0,
            "new_intent_name_buffer": "",
            "new_intent_examples_buffer": "",
            # Buffer keys for examples (max 10 intents)
        }
        
        for key, default_value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize buffer keys for each intent's examples (support up to 10)
        for i in range(10):
            buffer_key = f"multi_new_ex_buffer_{i}"
            if buffer_key not in st.session_state:
                st.session_state[buffer_key] = ""
        # ===== END SESSION STATE INITIALIZATION =====
        
        # Section 1: Model Status & Info
        st.markdown("### 📊 Model Status")
        model_info = training_editor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", model_info["model_status"])
        with col2:
            st.metric("Total Intents", model_info.get("intents_count", len(model_info["intents"])))
        with col3:
            if model_info.get("model_size_mb"):
                st.metric("Model Size", f"{model_info['model_size_mb']} MB")
        
        st.divider()
        
        # Section 2: Intent Validation
        is_valid, validation_issues = training_editor.validate_intent_data()
        
        if not is_valid and validation_issues:
            st.warning("⚠️ **Validation Issues Found:**")
            for issue in validation_issues:
                st.markdown(f"  - {issue}")
            st.divider()
        
        # Section 3: Intent Management
        st.markdown("### 🎯 Intent Management")
        
        # Get current intents
        intents = training_editor.load_intents()
        
        if intents:
            # Create tabs for each intent
            intent_names = [i["name"] for i in intents]
            
            # Sidebar for intent selection
            col_select, col_add = st.columns([2, 1])
            
            with col_select:
                selected_intent_idx = st.selectbox(
                    "Select Intent to Edit",
                    range(len(intents)),
                    format_func=lambda i: f"{intents[i]['name']} ({len(intents[i]['examples'])} examples)",
                    key="selected_intent_editor"
                )
            
            with col_add:
                if st.button("➕ New Intent", use_container_width=True):
                    st.session_state.show_new_intent_form = True
            
            st.divider()
            
            # Edit selected intent
            selected_intent = intents[selected_intent_idx]
            
            st.markdown(f"#### Intent: `{selected_intent['name']}`")
            st.markdown(f"**Examples: {len(selected_intent['examples'])}**")
            
            # Tab 1: View & Delete Examples
            st.markdown("**Current Training Examples:**")
            
            examples = selected_intent.get("examples", [])
            
            if examples:
                # Display examples with delete buttons
                for i, example in enumerate(examples):
                    col_ex, col_del = st.columns([5, 1])
                    
                    with col_ex:
                        st.text(f"  {i+1}. {example}")
                    
                    with col_del:
                        if st.button("🗑️", key=f"del_ex_{selected_intent_idx}_{i}", help="Delete this example"):
                            success, msg = training_editor.delete_example(selected_intent["name"], i)
                            if success:
                                st.success(msg)
                                st.session_state.training_changes = True
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                st.info("No examples yet. Add some below.")
            
            st.divider()
            
            # Add New Examples (multiline, one per line)
            st.markdown("**Add New Examples:**")
            add_examples_key = f"multi_new_ex_{selected_intent_idx}"
            add_examples_buffer = f"multi_new_ex_buffer_{selected_intent_idx}"
            
            # Initialize buffer session state if not exists
            if add_examples_buffer not in st.session_state:
                st.session_state[add_examples_buffer] = ""
            
            add_examples_text = st.text_area(
                "Enter examples (one per line):",
                value=st.session_state[add_examples_buffer],
                height=140,
                key=add_examples_key,
                placeholder="Example 1\nExample 2\nExample 3"
            )

            col_add_btn, col_clear_btn = st.columns([1, 1])
            with col_add_btn:
                if st.button("✅ Add Examples", use_container_width=True, key=f"add_multi_btn_{selected_intent_idx}"):
                    if add_examples_text and add_examples_text.strip():
                        success, msg = training_editor.add_example(selected_intent["name"], add_examples_text)
                        if success:
                            st.success(msg)
                            st.session_state.training_changes = True
                            # Clear the buffer key to reset the text area on next render
                            st.session_state[add_examples_buffer] = ""
                            st.rerun()
                        else:
                            st.warning(msg)
                    else:
                        st.error("Please enter one or more examples (one per line)")

            with col_clear_btn:
                if st.button("🧹 Clear All Examples", use_container_width=True, key=f"clear_all_ex_{selected_intent_idx}"):
                    confirm_key = f"confirm_clear_{selected_intent_idx}"
                    if st.session_state.get(confirm_key):
                        success, msg = training_editor.clear_examples(selected_intent["name"])
                        if success:
                            st.success(msg)
                            st.session_state.training_changes = True
                            st.session_state[confirm_key] = False
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("⚠️ Click again to confirm clearing all examples")
                        st.session_state[confirm_key] = True
            
            st.divider()
            
            # Tab 3: Intent Stats
            st.markdown("**Intent Statistics:**")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Example Count", len(examples))
            
            with col_stat2:
                avg_length = sum(len(ex.split()) for ex in examples) / len(examples) if examples else 0
                st.metric("Avg Words/Example", f"{avg_length:.1f}")
            
            with col_stat3:
                min_examples_ok = "✅" if len(examples) >= 5 else "⚠️"
                st.metric("Min Examples (5+)", f"{min_examples_ok}")
            
            st.divider()
            
            # Delete Intent (with double-confirmation)
            col_empty, col_del_intent = st.columns([4, 1])
            with col_del_intent:
                delete_intent_key = f"confirm_del_{selected_intent_idx}"
                
                if st.button("🗑️ Delete Intent", use_container_width=True, key=f"del_intent_btn_{selected_intent_idx}"):
                    if st.session_state.get(delete_intent_key):
                        # Second confirmation - actually delete
                        success, msg = training_editor.delete_intent(selected_intent["name"])
                        if success:
                            st.success(msg)
                            st.session_state.training_changes = True
                            # Clear confirmation flag
                            st.session_state[delete_intent_key] = False
                            st.rerun()
                        else:
                            st.error(msg)
                            st.session_state[delete_intent_key] = False
                    else:
                        # First confirmation - show warning
                        st.warning(f"⚠️ **Are you sure?** Click again to permanently delete '{selected_intent['name']}'")
                        st.session_state[delete_intent_key] = True
        else:
            st.info("No intents found. Create your first intent to get started.")
            if st.button("➕ Create First Intent", use_container_width=True, type="primary"):
                st.session_state.show_new_intent_form = True
                st.rerun()
        
        st.divider()
        
        # Section 4: Create New Intent
        if st.session_state.get("show_new_intent_form"):
            st.markdown("### ➕ Create New Intent")
            
            # Initialize buffer keys for the form inputs
            if "new_intent_name_buffer" not in st.session_state:
                st.session_state.new_intent_name_buffer = ""
            if "new_intent_examples_buffer" not in st.session_state:
                st.session_state.new_intent_examples_buffer = ""
            
            col_name, col_examples = st.columns([1, 2])
            
            with col_name:
                new_intent_name = st.text_input(
                    "Intent Name",
                    value=st.session_state.new_intent_name_buffer,
                    placeholder="e.g., check_balance",
                    key="new_intent_name_input"
                )
            
            with col_examples:
                new_examples_text = st.text_area(
                    "Initial Examples (one per line)",
                    value=st.session_state.new_intent_examples_buffer,
                    placeholder="What's my balance?\nShow my balance\nCheck balance",
                    height=100,
                    key="new_intent_examples_input"
                )
            
            col_create, col_cancel = st.columns(2)
            with col_create:
                if st.button("✅ Create Intent", use_container_width=True):
                    if new_intent_name.strip():
                        examples_list = [ex.strip() for ex in new_examples_text.split("\n") if ex.strip()]
                        success, msg = training_editor.create_intent(new_intent_name, examples_list)
                        if success:
                            st.success(msg)
                            st.session_state.show_new_intent_form = False
                            st.session_state.training_changes = True
                            # Clear buffers
                            st.session_state.new_intent_name_buffer = ""
                            st.session_state.new_intent_examples_buffer = ""
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please enter an intent name")
            
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_new_intent_form = False
                    # Clear buffers
                    st.session_state.new_intent_name_buffer = ""
                    st.session_state.new_intent_examples_buffer = ""
                    st.rerun()
        
        st.divider()
        
        # Section 5: Retrain Model
        st.markdown("### 🚀 Retrain NLU Model")
        st.markdown("Train the intent classification model with current intents and examples")
        
        # Pre-training validation
        intents = training_editor.load_intents()
        is_valid, validation_issues = training_editor.validate_intent_data()
        
        # Show validation status
        if not is_valid and validation_issues:
            st.warning("⚠️ **Training Readiness Check:**")
            for issue in validation_issues:
                st.markdown(f"  - {issue}")
            st.info("💡 **Recommendation:** Add more training examples for better model performance")
        else:
            st.success(f"✅ **Model Ready to Train:** {len(intents)} intents with {sum(len(i.get('examples', [])) for i in intents)} examples")
        
        st.divider()
        
        # Initialize session state for training parameters if not exists
        if 'epochs_value' not in st.session_state:
            st.session_state.epochs_value = 20
        if 'batch_size_value' not in st.session_state:
            st.session_state.batch_size_value = 16
        if 'learning_rate_value' not in st.session_state:
            st.session_state.learning_rate_value = 0.001
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
        
        # Training Parameters - Clean Single Row
        st.markdown("**🎯 Training Configuration**")
        
        col_epochs, col_batch, col_lr = st.columns(3)
        
        # Epochs Parameter
        with col_epochs:
            st.markdown('<div style="padding: 1rem; background-color: rgba(0, 217, 255, 0.05); border-radius: 0.5rem; border-left: 3px solid #00D9FF;">', unsafe_allow_html=True)
            st.markdown("**Epochs**")
            epochs_input = st.number_input(
                "Epochs",
                min_value=1,
                max_value=100,
                value=st.session_state.epochs_value,
                step=1,
                label_visibility="collapsed"
            )
            st.session_state.epochs_value = epochs_input
            st.caption("Training passes")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Batch Size Parameter
        with col_batch:
            st.markdown('<div style="padding: 1rem; background-color: rgba(0, 217, 255, 0.05); border-radius: 0.5rem; border-left: 3px solid #00D9FF;">', unsafe_allow_html=True)
            st.markdown("**Batch Size**")
            batch_input = st.number_input(
                "Batch Size",
                min_value=2,
                max_value=64,
                value=st.session_state.batch_size_value,
                step=1,
                label_visibility="collapsed"
            )
            st.session_state.batch_size_value = batch_input
            st.caption("Samples per batch")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Learning Rate Parameter
        with col_lr:
            st.markdown('<div style="padding: 1rem; background-color: rgba(0, 217, 255, 0.05); border-radius: 0.5rem; border-left: 3px solid #00D9FF;">', unsafe_allow_html=True)
            st.markdown("**Learning Rate**")
            lr_input = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=1.0,
                value=st.session_state.learning_rate_value,
                step=0.001,
                format="%.4f",
                label_visibility="collapsed"
            )
            st.session_state.learning_rate_value = lr_input
            st.caption("Step size (0.0001-1.0)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Helper Text Section
        st.markdown("**📌 Parameter Guide:**")
        col_helper1, col_helper2 = st.columns(2)
        with col_helper1:
            st.info("📈 **Higher epochs** improve accuracy but increase training time")
        with col_helper2:
            st.info("⚡ **Higher learning rate** speeds up learning but may cause instability")
        
        # Train Button - Centered Below
        col_empty1, col_btn, col_empty2 = st.columns([1, 2, 1])
        
        with col_btn:
            # Train button with validation checks
            can_train = len(intents) > 0 and is_valid and not st.session_state.training_in_progress
            
            if st.button(
                "🎓 Train Model", 
                use_container_width=True, 
                type="primary",
                disabled=not can_train,
                help="Requires at least one intent with 5+ examples" if not can_train else "Start model training"
            ):
                # Additional safety checks before training
                if not intents:
                    st.error("❌ No intents found. Create at least one intent with examples.")
                elif not is_valid:
                    st.error("❌ Validation failed. Please address the issues above.")
                else:
                    # Proceed with training
                    st.session_state.training_in_progress = True
                    
                    st.markdown("### 🔄 Training started… Streaming logs below")
                    
                    with st.spinner("🔄 Training model... This may take a moment..."):
                        # Calculate dropout from learning rate (as regularization indicator)
                        drop = 0.2 * (st.session_state.learning_rate_value / 0.001)  # Scale dropout based on LR
                        drop = min(drop, 0.5)  # Cap at max 0.5
                        
                        success, message, stats = training_editor.retrain_model(
                            n_iter=st.session_state.epochs_value,
                            batch_size=st.session_state.batch_size_value,
                            drop=drop,
                            learning_rate=st.session_state.learning_rate_value
                        )
                        
                        st.session_state.training_in_progress = False
                        
                        if success:
                            st.success(message)
                            st.session_state.training_changes = False
                            st.session_state.model_needs_reload = True
                            
                            # Display training stats in a professional format
                            st.markdown("**📊 Training Statistics:**")
                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            with col_s1:
                                st.metric("Intents", stats.get("intents", 0))
                            with col_s2:
                                st.metric("Examples", stats.get("examples", 0))
                            with col_s3:
                                st.metric("Epochs", stats.get("epochs", 0))
                            with col_s4:
                                st.metric("Final Loss", f"{stats.get('final_loss', 0):.4f}")
                            
                            st.success("✅ Model trained successfully! It will be used for the next prediction.")
                        else:
                            st.error(f"❌ Training failed: {message}")
        
        # Show reload button if model was retrained
        if st.session_state.get("model_needs_reload"):
            st.divider()
            if st.button("🔄 Reload Model (Update chatbot)", use_container_width=True, type="secondary"):
                success, msg = training_editor.reload_intent_model()
                if success:
                    st.success(msg)
                    st.session_state.model_needs_reload = False
                    st.rerun()
                else:
                    st.warning(msg)
        
        st.divider()
        
        # Section 6: Training Guidelines
        st.markdown("### 📖 Training Best Practices")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            **Do's:**
            - ✅ Use 5-10+ examples per intent
            - ✅ Vary phrasing and word order
            - ✅ Include real user queries
            - ✅ Check examples for quality
            - ✅ Train after adding examples
            """)
        
        with col_guide2:
            st.markdown("""
            **Don'ts:**
            - ❌ Mix multiple intents in one example
            - ❌ Use vague or generic examples
            - ❌ Have too few examples (< 5)
            - ❌ Forget to train after edits
            - ❌ Use exact same examples for different intents
            """)
    

    
    # ============================================
    # TAB 5: NLU VISUALIZER (NEW)
    # ============================================
    with tab_visualize:
        from nlu_engine import training_editor
        
        st.subheader("🧪 NLU Visualizer")
        st.markdown("Test the intent classification and entity extraction on sample queries")
        
        # Input section
        st.markdown("### 📝 Test Query")
        
        # Initialize session state buffers for test query (use buffer instead of direct widget key)
        if "nlu_test_query_buffer" not in st.session_state:
            st.session_state.nlu_test_query_buffer = ""
        
        test_query = st.text_input(
            "Enter a test query",
            value=st.session_state.nlu_test_query_buffer,
            placeholder="e.g., 'Transfer 5000 to account 1002'",
            key="nlu_test_query"
        )
        
        if test_query:
            col_predict, col_entity = st.columns(2)
            
            # Intent Prediction
            with col_predict:
                st.markdown("#### 🎯 Intent Prediction")
                
                with st.spinner("Analyzing intent..."):
                    intent_result = training_editor.predict_intent_with_confidence(test_query)
                
                if intent_result["success"]:
                    intent = intent_result["intent"]
                    confidence = intent_result["confidence"]
                    
                    # Display main prediction
                    st.markdown(f"**Predicted Intent:** `{intent}`")
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                    
                    # Color code confidence
                    if confidence >= 0.8:
                        st.success("✅ High confidence prediction")
                    elif confidence >= 0.6:
                        st.warning("⚠️ Medium confidence - may need more training")
                    else:
                        st.error("❌ Low confidence - intent unclear")
                    
                    # Show all scores
                    st.markdown("**All Intent Scores:**")
                    all_scores = intent_result.get("all_scores", {})
                    
                    if all_scores:
                        import pandas as pd
                        scores_df = pd.DataFrame([
                            {"Intent": intent, "Score": score}
                            for intent, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(scores_df, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        st.bar_chart(scores_df.set_index("Intent"))
                else:
                    st.error(f"❌ Error: {intent_result['error']}")
            
            # Entity Extraction
            with col_entity:
                st.markdown("#### 🏷️ Entity Extraction")
                
                with st.spinner("Extracting entities..."):
                    entity_result = training_editor.extract_entities_from_query(test_query)
                
                if entity_result["success"]:
                    entities = entity_result.get("entities", [])
                    
                    if entities:
                        st.markdown(f"**Found {len(entities)} entities:**")
                        
                        for entity in entities:
                            entity_type = entity.get("type", "UNKNOWN")
                            entity_value = entity.get("value", "")
                            
                            col_type, col_value = st.columns([1, 2])
                            with col_type:
                                st.markdown(f"`{entity_type}`")
                            with col_value:
                                st.markdown(f"**{entity_value}**")
                    else:
                        st.info("No entities detected in this query")
                else:
                    st.warning(f"⚠️ Entity extraction: {entity_result['error']}")
            
            st.divider()
            
            # Detailed Analysis
            st.markdown("### 📊 Detailed Analysis")
            
            # Reconstruction of query with predicted intent
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.markdown("**Query Breakdown:**")
                st.markdown(f"""
                - **Input:** {test_query}
                - **Token Count:** {len(test_query.split())}
                - **Character Count:** {len(test_query)}
                """)
            
            with col_analysis2:
                st.markdown("**Prediction Summary:**")
                if intent_result["success"]:
                    st.markdown(f"""
                    - **Intent:** {intent_result['intent']}
                    - **Confidence:** {intent_result['confidence']*100:.1f}%
                    - **Status:** {'✅ Ready' if intent_result['confidence'] >= 0.6 else '⚠️ Uncertain'}
                    """)
        else:
            st.info("💡 Enter a query above to test the NLU model")
        
        st.divider()
        
        # Test Examples
        st.markdown("### 🧪 Quick Test Examples")
        st.markdown("Click to test these predefined queries:")
        
        examples = [
            "What's my account balance?",
            "Transfer 5000 to account 1002",
            "Show my recent transactions",
            "Block my debit card",
            "Find nearest ATM"
        ]
        
        cols = st.columns(len(examples))
        for col, example in zip(cols, examples):
            with col:
                if st.button(f"🔍 \"{example[:15]}...\"", key=f"test_ex_{example}", use_container_width=True):
                    st.session_state.nlu_test_query_buffer = example
                    st.rerun()
        
        st.divider()
        
        # Model Validation Status
        st.markdown("### ✅ Model Validation")
        
        model_info = training_editor.get_model_info()
        is_valid, issues = training_editor.validate_intent_data()
        
        col_model_status, col_model_intents = st.columns(2)
        
        with col_model_status:
            st.markdown(f"**Model Status:** {model_info['model_status']}")
            if model_info["model_exists"]:
                st.success(f"Path: {model_info['model_path']}")
        
        with col_model_intents:
            st.markdown(f"**Intents:** {len(model_info['intents'])}")
            total_examples = sum(len(i.get("examples", [])) for i in model_info["intents"])
            st.markdown(f"**Training Examples:** {total_examples}")
        
        if is_valid:
            st.success("✅ Intent data is valid and ready for training")
        else:
            st.warning("⚠️ **Issues found in intent data:**")
            for issue in issues:
                st.markdown(f"  - {issue}")
    
    # ============================================
    # TAB 7: SYSTEM MANAGEMENT
    # ============================================
    with tab_system:
        st.subheader("⚙️ System Management & Administration")
        st.markdown("Database management, health checks, and system controls")
        
        col_system1, col_system2 = st.columns(2)
        
        with col_system1:
            st.markdown("### 📊 System Health")
            
            # Database status
            try:
                conn = get_conn()
                conn.close()
                st.success("✅ Database: Connected")
            except Exception as e:
                st.error(f"❌ Database: {str(e)}")
            
            # Model status
            try:
                from nlu_engine import training_editor
                model_info = training_editor.get_model_info()
                if model_info.get("model_exists"):
                    st.success(f"✅ Model: Ready ({model_info.get('model_size_mb', 0)} MB)")
                else:
                    st.warning("⚠️ Model: Not trained yet")
            except Exception:
                st.error("❌ Model: Check failed")
            
            # NLU Engine status
            try:
                from nlu_engine.infer_intent import _load_intent_model
                _load_intent_model()
                st.success("✅ NLU Engine: Running")
            except Exception:
                st.warning("⚠️ NLU Engine: Fallback mode (fuzzy matching)")
            
            st.divider()
            
            st.markdown("### 🗄️ Database Management")
            
            col_db1, col_db2 = st.columns(2)
            
            with col_db1:
                if st.button("📊 Database Stats", use_container_width=True):
                    try:
                        conn = get_conn()
                        cur = conn.cursor()
                        
                        # Get table stats
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = cur.fetchall()
                        st.markdown(f"**Tables:** {len(tables)}")
                        for table in tables:
                            st.markdown(f"  - {table[0]}")
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col_db2:
                if st.button("🗑️ Clear Logs", use_container_width=True):
                    with st.spinner("Clearing logs..."):
                        try:
                            nlu_logs.clear_nlu_logs()
                            st.success("✅ Logs cleared successfully")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with col_system2:
            st.markdown("### 🎛️ Admin Controls")
            
            st.markdown("**System Actions**")
            
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                if st.button("🔄 Reload Model", use_container_width=True):
                    try:
                        from nlu_engine import training_editor
                        success, msg = training_editor.reload_intent_model()
                        if success:
                            st.success(msg)
                        else:
                            st.warning(msg)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col_action2:
                if st.button("🔄 Refresh Cache", use_container_width=True):
                    st.cache_data.clear()
                    st.success("✅ Cache cleared")
            
            st.divider()
            
            st.markdown("**Configuration**")
            
            config_section = st.expander("View Configuration", expanded=False)
            with config_section:
                st.markdown("""
                **App Configuration:**
                - Python Version: 3.13+
                - Framework: Streamlit
                - Database: SQLite3
                - NLU: spaCy + Custom
                
                **Paths:**
                """)
                
                try:
                    from nlu_engine import training_editor
                    model_info = training_editor.get_model_info()
                    st.markdown(f"- Model: `{model_info.get('model_path')}`")
                    st.markdown(f"- Intents: `{model_info.get('intents_path')}`")
                except Exception:
                    pass
        
        st.divider()
        
        st.markdown("### 📋 Maintenance Tasks")
        
        col_maint1, col_maint2, col_maint3 = st.columns(3)
        
        with col_maint1:
            if st.button("🧹 Optimize Database", use_container_width=True):
                with st.spinner("Optimizing..."):
                    try:
                        conn = get_conn()
                        conn.execute("VACUUM")
                        conn.close()
                        st.success("✅ Database optimized")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col_maint2:
            if st.button("📊 Export Database", use_container_width=True):
                st.info("Export feature coming soon...")
        
        with col_maint3:
            if st.button("📈 Generate Report", use_container_width=True):
                st.info("Report generation coming soon...")
    
    # ============================================
    # TAB 8: KNOWLEDGE BASE
    # ============================================
    with tab_kb:
        st.subheader("📖 Knowledge Base Manager")
        st.markdown("Manage FAQs, references, and knowledge content for the chatbot")
        
        kb_path = Path(__file__).parent / "kb.md"
        
        if kb_path.exists():
            kb_text = kb_path.read_text(encoding="utf-8")
        else:
            kb_text = """# Knowledge Base

## Banking FAQs

### Account Balance
- Check balance: "What's my balance?"
- View account details: "Show account info"

### Money Transfer
- Transfer steps: Account number → Amount → Transaction PIN
- Daily limit: Check your account type

### ATM Locator
- Find nearest ATM: "Where's the nearest ATM?"
- 24/7 availability

### Card Management
- Block/Unblock debit or credit card
- Report lost card

---

## Chatbot Capabilities
- Intent Recognition: Understand user intent from natural language
- Entity Extraction: Extract account numbers, amounts, dates
- Dialogue Management: Multi-turn conversations
- Banking Operations: Check balance, transfer, transactions

---

Edit this KB to add FAQs and reference material.
"""
        
        # KB Controls & Stats
        st.markdown("### 📊 Knowledge Base Statistics")
        
        col_kb_stat1, col_kb_stat2, col_kb_stat3, col_kb_stat4 = st.columns(4)
        
        with col_kb_stat1:
            kb_lines = len(kb_text.split('\n'))
            st.metric("Total Lines", kb_lines)
        
        with col_kb_stat2:
            kb_sections = kb_text.count('##')
            st.metric("Sections", kb_sections)
        
        with col_kb_stat3:
            kb_size = len(kb_text.encode('utf-8')) / 1024
            st.metric("Size (KB)", f"{kb_size:.2f}")
        
        with col_kb_stat4:
            kb_words = len(kb_text.split())
            st.metric("Total Words", kb_words)
        
        st.divider()
        
        # KB Tabs
        kb_tab1, kb_tab2, kb_tab3, kb_tab4 = st.tabs([
            "✏️ Editor",
            "📈 Analytics",
            "🔍 Search",
            "📋 FAQ Manager"
        ])
        
        # ====== KB TAB 1: EDITOR ======
        with kb_tab1:
            st.markdown("#### Edit Knowledge Base")
            
            col_editor_main, col_editor_tools = st.columns([3, 1])
            
            with col_editor_main:
                new_kb = st.text_area(
                    "Knowledge Base Content (Markdown)",
                    value=kb_text,
                    height=400,
                    help="Use markdown format. This is available to the chatbot for reference."
                )
            
            with col_editor_tools:
                st.markdown("**Quick Templates**")
                
                if st.button("➕ Add FAQ", use_container_width=True):
                    new_kb += "\n\n### New FAQ\n- Question: ...\n- Answer: ..."
                
                if st.button("➕ Add Section", use_container_width=True):
                    new_kb += "\n\n## New Section\n\nContent here..."
                
                st.divider()
                
                st.markdown("**Editor Tools**")
                
                if st.button("🗑️ Clear", use_container_width=True):
                    if st.session_state.get("confirm_clear_kb"):
                        new_kb = ""
                        st.success("Cleared!")
                        st.session_state["confirm_clear_kb"] = False
                    else:
                        st.warning("Click again to confirm clear")
                        st.session_state["confirm_clear_kb"] = True
            
            st.divider()
            
            # Save & Preview
            col_save, col_preview = st.columns([1, 2])
            
            with col_save:
                if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                    try:
                        kb_path.write_text(new_kb, encoding="utf-8")
                        st.success("✅ Knowledge base updated successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to save: {e}")
                
                st.divider()
                
                st.markdown("**File Options**")
                
                # Export
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        "📥 Export MD",
                        data=new_kb,
                        file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col_exp2:
                    import json
                    kb_json = json.dumps({"content": new_kb, "updated": datetime.now().isoformat()})
                    st.download_button(
                        "📥 Export JSON",
                        data=kb_json,
                        file_name=f"kb_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col_preview:
                st.markdown("#### Live Preview")
                st.markdown(new_kb)
        
        # ====== KB TAB 2: ANALYTICS ======
        with kb_tab2:
            st.markdown("#### Knowledge Base Usage Analytics")
            
            col_analytics1, col_analytics2 = st.columns(2)
            
            with col_analytics1:
                st.markdown("**Content Distribution**")
                
                # Parse KB sections
                sections = [line for line in kb_text.split('\n') if line.startswith('##')]
                if sections:
                    section_names = [s.replace('##', '').strip() for s in sections]
                    section_counts = [len(line) for line in kb_text.split('\n')]
                    
                    st.markdown("""
                    **Sections Overview:**
                    """)
                    for section in section_names:
                        st.markdown(f"- {section}")
                
                st.divider()
                
                st.markdown("**Content Quality Metrics**")
                
                # Calculate readability scores
                avg_line_length = len(kb_text) / max(1, len(kb_text.split('\n')))
                bullet_points = kb_text.count('-')
                code_blocks = kb_text.count('```')
                
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("Avg Line Length", f"{avg_line_length:.0f} chars")
                with col_metric2:
                    st.metric("Bullet Points", bullet_points)
                with col_metric3:
                    st.metric("Code Blocks", code_blocks)
            
            with col_analytics2:
                st.markdown("**Query Usage Over Time**")
                
                # Generate sample line chart data
                import pandas as pd
                import numpy as np
                
                days = pd.date_range(end=datetime.now(), periods=30, freq='D')
                # Simulated query counts trending upward
                queries = np.cumsum(np.random.randint(5, 15, 30))
                
                df_usage = pd.DataFrame({
                    "Date": days,
                    "Queries": queries,
                    "KB Hits": queries * 0.7
                })
                
                st.line_chart(df_usage.set_index("Date")[["Queries", "KB Hits"]])
                
                st.divider()
                
                st.markdown("**Popular Topics**")
                
                # Simulate topic popularity
                topics_data = {
                    "Account Balance": 45,
                    "Money Transfer": 38,
                    "Card Management": 22,
                    "ATM Locator": 18,
                    "Transaction History": 15
                }
                
                df_topics = pd.DataFrame(list(topics_data.items()), columns=["Topic", "References"])
                df_topics = df_topics.sort_values("References", ascending=True)
                
                st.bar_chart(df_topics.set_index("Topic"))
        
        # ====== KB TAB 3: SEARCH ======
        with kb_tab3:
            st.markdown("#### Search Knowledge Base")
            
            search_query = st.text_input(
                "Search KB content",
                placeholder="e.g., 'balance', 'transfer', 'card'",
                key="kb_search"
            )
            
            if search_query:
                # Simple search
                lines = kb_text.split('\n')
                results = [line for line in lines if search_query.lower() in line.lower()]
                
                if results:
                    st.markdown(f"**Found {len(results)} matches:**")
                    st.divider()
                    
                    for idx, result in enumerate(results[:20], 1):
                        if result.strip():
                            st.markdown(f"{idx}. {result}")
                else:
                    st.warning(f"No matches found for '{search_query}'")
            else:
                st.info("💡 Enter a search term to find content in the knowledge base")
        
        # ====== KB TAB 4: FAQ MANAGER ======
        with kb_tab4:
            st.markdown("#### Frequently Asked Questions Manager")
            
            faq_mode = st.radio("Choose action", ["View FAQs", "Add FAQ", "Remove FAQ"], horizontal=True)
            
            if faq_mode == "View FAQs":
                st.markdown("**Current FAQs in Knowledge Base**")
                
                # Extract FAQs
                faq_lines = [line for line in kb_text.split('\n') if line.startswith('###')]
                
                if faq_lines:
                    for idx, faq in enumerate(faq_lines, 1):
                        faq_title = faq.replace('###', '').strip()
                        st.markdown(f"**{idx}. {faq_title}**")
                else:
                    st.info("No FAQs defined yet. Add some in the Editor tab!")
            
            elif faq_mode == "Add FAQ":
                st.markdown("**Create New FAQ**")
                
                col_faq1, col_faq2 = st.columns(2)
                
                with col_faq1:
                    faq_title = st.text_input("FAQ Title", placeholder="e.g., How to reset PIN?")
                
                with col_faq2:
                    faq_category = st.selectbox(
                        "Category",
                        ["General", "Accounts", "Transfers", "Cards", "ATM", "Other"]
                    )
                
                faq_answer = st.text_area(
                    "Answer",
                    placeholder="Provide detailed answer...",
                    height=150
                )
                
                if st.button("➕ Add FAQ", use_container_width=True, type="primary"):
                    if faq_title and faq_answer:
                        new_faq = f"\n\n### {faq_title}\n{faq_answer}"
                        updated_kb = kb_text + new_faq
                        
                        try:
                            kb_path.write_text(updated_kb, encoding="utf-8")
                            st.success(f"✅ FAQ added: {faq_title}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.error("Please fill in title and answer")
            
            elif faq_mode == "Remove FAQ":
                st.markdown("**Remove FAQ**")
                
                faq_lines = [line for line in kb_text.split('\n') if line.startswith('###')]
                faq_titles = [faq.replace('###', '').strip() for faq in faq_lines]
                
                if faq_titles:
                    selected_faq = st.selectbox("Select FAQ to remove", faq_titles)
                    
                    if st.button("🗑️ Delete FAQ", use_container_width=True, type="secondary"):
                        # Find and remove the FAQ
                        lines = kb_text.split('\n')
                        new_lines = []
                        skip_until_next_section = False
                        
                        for line in lines:
                            if line.startswith(f"### {selected_faq}"):
                                skip_until_next_section = True
                                continue
                            elif skip_until_next_section and (line.startswith('##') or line.startswith('###')):
                                skip_until_next_section = False
                            
                            if not skip_until_next_section:
                                new_lines.append(line)
                        
                        updated_kb = '\n'.join(new_lines)
                        
                        try:
                            kb_path.write_text(updated_kb, encoding="utf-8")
                            st.success(f"✅ FAQ removed: {selected_faq}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.info("No FAQs to remove")

elif selected == "❓ Help":
    st.markdown("# ❓ Help & Documentation")
    st.markdown("Learn how to use BankBot AI and get started quickly")
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 Chatbot Commands")
        st.markdown("""
        Try these natural language queries with the chatbot:
        
        **Balance Check:**
        - "Check balance for 1001"
        - "What's my balance?"
        - "Show balance account 1002"
        
        **Money Transfer:**
        - "Transfer 5000 to account 1002"
        - "Send ₹2000 from 1001 to 1003"
        - "I want to transfer money"
        
        **ATM & Branches:**
        - "Find ATM"
        - "Nearby ATM"
        - "ATM near me"
        
        **Transaction History:**
        - "Show transactions for 1001"
        - "What are my recent transactions?"
        - "Transaction history for account 1002"
        
        **Card Management:**
        - "Block my debit card"
        - "Block credit card for account 1001"
        - "Unblock my card"
        
        **General:**
        - "Hello" or "Hi"
        - "Help"
        - "What can you do?"
        """)
    
    with col2:
        st.markdown("### ⚙️ System Information")
        st.markdown("""
        **BankBot AI Features:**
        - ✅ Intent Recognition (spaCy + fuzzy matching)
        - ✅ Entity Extraction (regex + NER)
        - ✅ Dialogue Management (slot filling)
        - ✅ Database Integration (SQLite)
        - ✅ Session Memory
        - ✅ Transaction PIN Security
        
        **Security:**
        - 🔐 Login PIN: Authenticate to account
        - 🔐 Transaction PIN: Verify money transfers
        - Both PINs are hashed using bcrypt
        
        **Demo Accounts:**
        | Account | Name | Login PIN | Trans PIN |
        |---------|------|-----------|-----------|
        | 1001 | Rajesh Kumar | 1234 | 4321 |
        | 1002 | Priya Sharma | 5678 | 8765 |
        | 1003 | Amit Patel   | 9012 | 2109 |
        
        **Architecture:**
        - Frontend: Streamlit
        - NLU: spaCy + custom extractors
        - Database: SQLite3
        - Security: bcrypt (PIN hashing)
        """)

elif selected == "📞 Support":
    st.markdown("# 📞 Support & Feedback")
    st.markdown("Get help, report issues, or send feedback")
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🆘 Getting Help")
        st.markdown("""
        **If you're having trouble:**
        
        1. **Check the Help page** ❓
           - Browse common chatbot commands
           - Review system features
           - Understand security setup
        
        2. **Use the Admin Panel** 🧾
           - View analytics and logs
           - Check recent chat activity
           - Review knowledge base
        
        3. **Try the Chatbot** 💬
           - Ask natural language questions
           - Test with demo accounts
           - Explore different intents
        
        4. **Review Documentation**
           - Check README.md for overview
           - See ADMIN_PANEL_GUIDE.md for features
           - Review troubleshooting guides
        """)
    
    with col2:
        st.markdown("### 📧 Feedback & Issues")
        st.markdown("""
        **Report an issue:**
        - Describe what went wrong
        - Include the error message
        - Check logs in Admin Panel
        - Provide account number if relevant
        
        **Suggest a feature:**
        - Describe the feature
        - Explain the benefit
        - Provide examples
        
        **Technical Support:**
        - Check database connectivity
        - Verify PIN entry (case-sensitive)
        - Ensure spaCy model installed
        - Check file permissions
        
        **Common Issues:**
        - "No logs showing" → Send queries first
        - "Database locked" → Close other connections
        - "Model not found" → Run model training
        - "Import error" → Reinstall packages
        """)
    
    st.divider()
    
    st.markdown("### 💡 Tips & Best Practices")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **For Chatbot:**
        - Use natural language
        - Be specific with amounts
        - Include account numbers
        - Verify responses
        """)
    
    with col2:
        st.success("""
        **For Admin Panel:**
        - Check metrics regularly
        - Monitor confidence scores
        - Review low-confidence queries
        - Track intent distribution
        """)
    
    with col3:
        st.warning("""
        **For Security:**
        - Don't share PINs
        - Logout when done
        - Use strong PINs
        - Verify amounts before confirming
        """)




