
# 🏦 BankBot AI – Intelligent Banking Chatbot

BankBot AI is an AI-powered banking chatbot built using **Python, Streamlit, NLU, and SQLite**.  
It helps users perform banking operations such as checking balance, transferring money, blocking cards, and also answers **general (non-banking) questions** using an LLM.

The project is developed milestone-wise to demonstrate **NLU, dialogue management, database integration, and admin analytics**.

---

## 🚀 Features

### 👤 User Features
- Login & Create Account
- Check account balance
- Transfer money securely using Transaction PIN
- Block debit/credit card
- View transaction history
- Find nearby ATMs
- Ask general questions (LLM-powered)

### 🧠 AI & NLU Features
- Intent recognition
- Entity extraction
- Slot filling dialogue management
- Strict transfer flow validation
- Banking vs Non-banking domain routing

### 🛠️ Admin Features (Milestone 4)
- View chatbot usage analytics
- Monitor intents & confidence scores
- Edit training data (intents & examples)
- Retrain NLU model
- View logs & errors
- FAQ & Knowledge Base management

---

## 🧩 Project Architecture

```

BankBot_AI/
│
├── main_app.py                # Chatbot UI + Admin Panel (Streamlit)
│
├── nlu_engine/
│   ├── **init**.py
│   ├── infer_intent.py        # Intent prediction
│   ├── intent_classifier.py
│   ├── entity_extractor.py    # Entity extraction logic
│   ├── domain_gate.py         # Banking / Non-banking routing
│   ├── nlu_router.py          # Dialogue & slot management
│   ├── train_intent.py        # Model training
│   ├── training_editor.py    # Admin training editor
│   ├── intents.json           # Intent training data
│   └── entities.json          # Entity definitions
│
├── database/
│   ├── db.py                  # SQLite connection
│   ├── bank_crud.py           # Banking operations
│   └── security.py            # PIN hashing & verification
│
├── experiments/
│   └── llm_groq.py            # LLM integration for non-banking queries
│
├── tests/                     # Unit & flow tests
│
├── README.md
└── requirements.txt



## 📌 Milestones Overview

### ✅ Milestone 1 – Chatbot Foundation
- Streamlit UI
- Basic chatbot flow
- User login & session handling

### ✅ Milestone 2 – NLU & Banking Logic
- Intent classification
- Entity extraction
- Slot filling dialogue
- Banking operations (balance, transfer, block card)

### ✅ Milestone 3 – Database & LLM Integration
- SQLite database
- Transaction history
- Secure PIN verification
- Non-banking query handling via LLM

### ✅ Milestone 4 – Admin Panel & Analytics
- Admin dashboard
- Training data editor
- NLU retraining controls
- Analytics & logs
- FAQ / Knowledge base

---

## 🔐 Security

- Login PIN & Transaction PIN stored using hashing
- Transaction PIN required only at final step
- Strict validation for transfer flow
- Session-based context management

---

## 🧪 Example Chat Flow

```

User: transfer money
Bot: Please provide your account number
User: 1001
Bot: Please provide receiver account number
User: 1002
Bot: How much would you like to transfer?
User: 5000
Bot: Please enter your transaction PIN
User: 4321
Bot: ✅ Transfer Successful

````

---

## ⚙️ Installation & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main_app.py
````



## 📈 Future Scope

* Voice-based chatbot
* Multi-language support
* Fraud detection using ML
* Integration with real banking APIs
* Mobile app version
* Role-based admin access

---

## 🧑‍💻 Technologies Used

* Python
* Streamlit
* SQLite
* spaCy / Custom NLU
* LLM (Groq / OpenAI compatible)
* bcrypt (security)

---

## 📌 Conclusion

BankBot AI demonstrates how AI, NLU, and databases can be combined to build a **secure, intelligent banking assistant**.
The milestone-based approach makes it scalable, explainable, and suitable for academic and real-world applications.

---

## 👨‍🎓 Developed By

**[VIBHEESHAN N K ]**
BankBot AI – Intelligent Banking Chatbot  
```
