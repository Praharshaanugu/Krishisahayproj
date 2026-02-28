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

# ---------- API Key Setup ----------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not configured.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------- Load Embedding Model ----------
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------- Load FAISS Index ----------
index = faiss.read_index("data/faiss_index/krishi_index.faiss")

# ---------- Load Chunks ----------
all_chunks = []
all_categories = []

chunk_base = Path("data/chunks")

for category in chunk_base.iterdir():
    if category.is_dir():
        for json_file in category.glob("*.json"):
            chunks = json.loads(json_file.read_text(encoding="utf-8"))
            for chunk in chunks:
                all_chunks.append(chunk)
                all_categories.append(category.name.lower())

# ---------- Crop Detection ----------
def detect_crop(query):
    crops = list(set(all_categories))
    for crop in crops:
        if crop in query.lower():
            return crop
    return None

# ---------- UI ----------
st.title("🌾 KrishiSahay")
st.caption("AI Agricultural Field Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your farming question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # -------- Embed Query --------
            query_embedding = embed_model.encode([query]).astype("float32")

            # -------- Crop Detection --------
            detected_crop = detect_crop(query)

            # -------- Retrieval --------
            distances, indices = index.search(query_embedding, 7)

            retrieved_chunks = []

            for i in indices[0]:
                if detected_crop:
                    if all_categories[i] == detected_crop:
                        retrieved_chunks.append(all_chunks[i])
                else:
                    retrieved_chunks.append(all_chunks[i])

            if len(retrieved_chunks) < 2:
                retrieved_chunks = [all_chunks[i] for i in indices[0]]

            retrieved_text = "\n\n".join(retrieved_chunks[:4])

            # -------- Scheme Intent Detection --------
            scheme_keywords = [
                "scheme", "subsidy", "loan", "pm kisan",
                "insurance", "benefit", "eligibility",
                "government", "yojana"
            ]

            is_scheme_query = any(word in query.lower() for word in scheme_keywords)

            # -------- Dynamic Prompt --------
            if is_scheme_query:
                prompt = f"""
You are an agricultural government scheme advisor.

Answer clearly about schemes, eligibility, benefits and how to apply.

Rules:
- Maximum 6 lines
- Simple farmer-friendly language
- Mention eligibility
- Mention benefit amount if available
- Mention how to apply

Knowledge:
{retrieved_text}

Farmer Question:
{query}

Answer:
"""
            else:
                prompt = f"""
You are an experienced agricultural field officer helping farmers.

Use the provided knowledge as primary reference.
You may use general agricultural knowledge if needed.

Answer format:
1. Likely cause
2. What to check
3. What to do immediately
4. When to monitor again

Rules:
- Maximum 6 lines
- Simple farmer-friendly language
- No technical jargon

Knowledge:
{retrieved_text}

Farmer Question:
{query}

Answer:
"""

            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )
                final_answer = response.text.strip()
            except Exception as e:
                final_answer = f"LLM Error: {str(e)}"

            st.write(final_answer)
            st.caption("⚠ Advice based on agricultural documents.")

            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )