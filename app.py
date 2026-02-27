import streamlit as st
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv

# ---------- Page Config ----------
st.set_page_config(page_title="KrishiSahay 🌾", layout="centered")

# ---------- API Key Setup (Cloud + Local Compatible) ----------
try:
    api_key = st.secrets["GEMINI_API_KEY"]  # Streamlit Cloud
except:
    load_dotenv()  # Local
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found. Configure secrets (Cloud) or .env file (Local).")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------- Load Models ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss_index/krishi_index.faiss")

with open("data/faiss_index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

all_chunks = []
chunk_base = Path("data/chunks")
for category in chunk_base.iterdir():
    if category.is_dir():
        for json_file in category.glob("*.json"):
            chunks = json.loads(json_file.read_text(encoding="utf-8"))
            all_chunks.extend(chunks)

# ---------- UI ----------
st.title("🌾 KrishiSahay")
st.caption("AI Agricultural Field Assistant")

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Telugu"]
)

# ---------- Translation Function ----------
def translate_text(text, target_language):
    if target_language == "English":
        return text

    prompt = f"""
Translate the following text into {target_language}.
Return only the translated text.

{text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text.strip()

# ---------- Session Memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Chat Input ----------
query = st.chat_input("Ask your farming question...")

if query:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # 🔹 Step 1: Translate user query to English
            internal_query = translate_text(query, "English")

            # 🔹 Step 2: Embed translated query
            query_embedding = embed_model.encode([internal_query]).astype("float32")

            # 🔹 Step 3: Retrieve relevant chunks
            distances, indices = index.search(query_embedding, 3)
            retrieved_text = "\n\n".join([all_chunks[i] for i in indices[0]])

            # 🔹 Step 4: Build grounded prompt
            prompt = f"""
You are KrishiSahay, a practical agricultural advisor.

Rules:
- Short answer (max 6 lines)
- Simple farmer-friendly language
- Give practical actions
- No technical jargon

Use only this knowledge:
{retrieved_text}

Farmer Question:
{internal_query}
"""

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            english_answer = response.text.strip()

            # 🔹 Step 5: Translate back to selected language
            final_answer = translate_text(english_answer, language)

            st.write(final_answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )