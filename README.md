🌾 KrishiSahay

AI-Powered Agricultural Assistant using Retrieval-Augmented Generation (RAG)

KrishiSahay is an intelligent agricultural advisory system designed to assist farmers with crop issues, pest management, soil health, fertilizer guidance, and government scheme information.

The system uses a Retrieval-Augmented Generation (RAG) architecture to provide grounded, reliable, and practical responses instead of generic AI answers.

🚀 Live Demo



🌍 Problem Statement

Many farmers lack easy access to agricultural experts and extension officers.
Existing digital solutions are often:

English-only

Generic chatbots

Not grounded in real agricultural knowledge

Dependent on constant internet connectivity

KrishiSahay addresses these challenges by combining a structured agricultural knowledge base with AI reasoning.

🧠 System Architecture

Farmer Query
↓
Sentence Embedding (MiniLM)
↓
FAISS Vector Search
↓
Relevant Knowledge Retrieval
↓
Gemini LLM Reasoning
↓
Farmer-Friendly Response

🏗 Tech Stack

Python

Streamlit – UI Layer

Sentence Transformers – Text Embeddings

FAISS – Vector Search Engine

Google Gemini API – Language Reasoning

RAG Architecture – Grounded Answer Generation

✨ Key Features

🌾 Crop advisory support

🐛 Pest and disease guidance

🧪 Soil and fertilizer recommendations

🏛 Government scheme awareness

🌐 Multilingual capability (English / Hindi / Telugu)

💬 Chat-based user interface

🔎 Grounded responses using local agricultural data

⚡ Fast semantic retrieval using FAISS

📂 Project Structure
KrishiSahay/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── chunks/
│   ├── cleaned/
│   └── faiss_index/
│
├── scripts/
│   ├── extract_text.py
│   ├── clean_text.py
│   ├── chunk_text.py
│   └── build_index.py
⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/KrishiSahay.git
cd KrishiSahay
2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   (Windows)
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add API Key

Create a .env file in the root directory:

GEMINI_API_KEY=your_api_key_herehttps://github.com/Praharshaanugu/Krishisahayproj.git
(Do not commit this file to GitHub.)

5️⃣ Run the Application
streamlit run app.py
🔬 How It Works

Agricultural documents are collected and cleaned.

Content is split into smaller chunks.

Each chunk is converted into vector embeddings.

FAISS stores embeddings for semantic search.

When a farmer asks a question:

The system retrieves the most relevant chunks.

Gemini generates a concise, farmer-friendly response based only on retrieved knowledge.

The response is displayed in a chat interface.

🛡 Security Practices

API keys are stored using environment variables.

.env file is excluded using .gitignore.

No sensitive credentials are committed to the repository.

🌱 Future Enhancements

Crop image disease detection

Offline LLM fallback mode

Hybrid keyword + vector search

Farmer eligibility engine for schemes

Voice input support

Deployment scaling
