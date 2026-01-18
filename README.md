

# 🏦 BankBot AI – Intelligent Banking Chatbot

BankBot AI is an **AI-powered banking chatbot** built using **Python, Streamlit, custom NLU, and SQLite**.
It allows users to perform core banking operations using **natural language**, while also providing a powerful **admin panel** for analytics, training, and monitoring.

The project is developed **milestone-wise** to clearly demonstrate:

* Natural Language Understanding (NLU)
* Dialogue & slot-filling management
* Secure banking workflows
* Database integration
* Admin analytics & retraining

In addition to banking tasks, BankBot AI can answer **general (non-banking) questions** using an **LLM**.
It helps users perform banking operations such as checking balance, transferring money, blocking cards, and also answers **general (non-banking) questions** using an LLM.

---

## 🚀 Features

### 👤 User Features

* User login & account creation
* Check account balance
* Transfer money securely using **Transaction PIN**
* Block debit / credit cards
* View transaction history
* Find nearby ATMs
* Ask general (non-banking) questions via **LLM**

---

### 🧠 AI & NLU Features

* Intent recognition
* Entity extraction (account numbers, amount, PIN, etc.)
* Slot-filling dialogue management
* Multi-turn conversation handling
* Strict validation for sensitive flows (money transfer)
* Banking vs Non-banking **domain routing**

---

### 🛠️ Admin Features (Milestone 4)

* Admin dashboard with usage analytics
* Monitor intents, confidence scores, and performance
* Training editor (add / edit intents & examples)
* Retrain NLU model from UI
* Logs & error monitoring
* FAQ & Knowledge Base management

---

## 🧩 Project Architecture (Complete Guide)

```
BankBot_AI/
│
├── main_app.py
│   └── Main Streamlit application
│       - Chatbot UI
│       - Admin panel
│       - Analytics dashboards
│
├── streamlit_app.py (optional)
│   └── NLU testing & visualization
│
├── database/
│   ├── db.py                # SQLite connection
│   ├── bank_crud.py         # Balance, transfer, transactions
│   ├── admin_analytics.py   # Usage & performance stats
│   ├── nlu_logs.py          # Intent & confidence logs
│   ├── llm_tracking.py      # LLM usage tracking
│   ├── security.py          # PIN hashing & verification
│   └── __init__.py
│
├── dialogue_manager/
│   └── dialogue_handler.py
│       - Multi-turn & slot-filling logic
│
├── nlu_engine/
│   ├── __init__.py
│   ├── domain_gate.py       # Banking vs non-banking routing
│   ├── infer_intent.py      # Intent prediction
│   ├── intent_classifier.py# ML intent model
│   ├── entity_extractor.py # Entity extraction
│   ├── nlu_router.py        # Intent → action mapping
│   ├── train_intent.py      # Model training
│   ├── training_editor.py  # Admin training editor
│   ├── intents.json        # Intent training data
│   └── entities.json       # Entity definitions
│
├── models/
│   ├── intent_model/        # Trained NLU models
│   ├── spacy_nlp/           # NLP assets
│   ├── backups/             # Model backups
│   └── training.log         # Training logs
│
├── experiments/
│   ├── llm_groq.py          # Groq LLM integration
│   ├── llm_handler.py       # LLM request handler
│   └── llm_local.py         # Local fallback logic
│
├── bankbot.db               # SQLite database
├── requirements.txt         # Python dependencies
├── README.md
└── .env                     # Environment variables
```

---

## 🔐 Security

* Login PIN & Transaction PIN stored using **hashed encryption**
* Transaction PIN requested only at the final step
* Strict validation for sensitive workflows
* Session-based context & state management

---

## 🧪 Example Chat Flow

```
User: Transfer money
Bot: Please provide your account number
User: 1001
Bot: Please provide receiver account number
User: 1002
Bot: How much would you like to transfer?
User: 5000
Bot: Please enter your transaction PIN
User: 4321
Bot: ✅ Transfer Successful
```

---

## ⚙️ Installation & Run (Step-by-Step)

### ✅ Prerequisites

* Python 3.9+
* Git
* Virtual environment (venv)

Check Python version:

```bash
python --version
```

---

### 📥 Step 1: Clone Repository

```bash
git clone https://github.com/<your-username>/BankBot-AI.git
cd BankBot-AI
```

---

### 🧪 Step 2: Create & Activate Virtual Environment

**Windows**

```bash
python -m venv bankbot_ai
bankbot_ai\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv bankbot_ai
source bankbot_ai/bin/activate
```

---

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
This installs:

Streamlit

SQLite support

NLP / ML libraries

LLM client libraries

Security & hashing utilities
---

### 🔐 Step 4: Configure `.env` File

Create a `.env` file in the project root.

Example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=optional_openai_key
APP_ENV=development
```

⚠️ **Do not commit `.env` to GitHub**
(It is already included in `.gitignore`)

---

### 🧠 Step 5: Prepare NLU Data (Optional)

Edit:

* `nlu_engine/intents.json`
* `nlu_engine/entities.json`

Example:

```json
{
  "check_balance": [
    "check my balance",
    "how much money do I have"
  ]
}
```

---

### 🚀 Step 6: Run the Application

```bash
streamlit run main_app.py
```

Open browser:

```
http://localhost:8501
```

---

## 🧑‍💼 Admin Panel

* Open **Admin** from sidebar
* View analytics & logs
* Edit intents & examples
* Retrain NLU model
* Manage FAQs / Knowledge Base

---

## 🔁 NLU Model Retraining

From Admin Panel:

* Set Epochs
* Set Batch Size
* Set Learning Rate
* Click **Train Model**

Model saved in:

```
models/intent_model/
```

---

## 🗄️ Database

* **Type:** SQLite
* **File:** `bankbot.db`
* Auto-created on first run
* Stores users, transactions, logs & analytics

---


✅ Quick Run Checklist

✔ Virtual environment activated
✔ Dependencies installed
✔ .env configured
✔ streamlit run main_app.py executed


---

## 📈 Future Scope

* Voice-based chatbot
* Multi-language support
* Fraud detection using ML
* Integration with real banking APIs
* Mobile application
* Role-based admin access

---

## 🧑‍💻 Technologies Used

* Python
* Streamlit
* SQLite
* Custom NLU / ML models
* LLM (Groq / OpenAI compatible)
* bcrypt (security)

---

## 📌 Conclusion

BankBot AI demonstrates how **AI, NLU, and databases** can be combined to build a **secure, intelligent banking assistant**.
Its **modular and milestone-based architecture** makes it scalable, explainable, and suitable for **academic projects, hackathons, and real-world applications**.

---

## 👨‍🎓 Developed By

**Vibheeshan N K**
*BankBot AI – Intelligent Banking Chatbot*

---

