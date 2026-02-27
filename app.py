import streamlit as st
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os

# ---------- Setup ----------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
st.set_page_config(page_title="KrishiSahay 🌾", layout="centered")
st.title("🌾 KrishiSahay")
st.caption("AI Agricultural Field Assistant")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
query = st.chat_input("Ask your farming question...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # Retrieve knowledge
            query_embedding = embed_model.encode([query]).astype("float32")
            distances, indices = index.search(query_embedding, 3)
            retrieved_text = "\n\n".join([all_chunks[i] for i in indices[0]])

            prompt = f"""
You are KrishiSahay, a practical agricultural advisor.

Rules:
- Short answer (6 lines max)
- Simple language
- Practical actions
- No technical jargon

Use only this knowledge:
{retrieved_text}

Farmer Question:
{query}
"""

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            answer = response.text
            st.write(answer)

            # Save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )