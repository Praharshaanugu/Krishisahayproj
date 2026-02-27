import streamlit as st
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv

# ---------- Page Config ----------
st.set_page_config(page_title="KrishiSahay 🌾", layout="centered")

# ---------- HF Token Setup ----------
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("HuggingFace token not configured.")
    st.stop()

# ---------- HuggingFace API Setup ----------
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_hf(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return str(result)

# ---------- Load Embedding Model ----------
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

# ---------- UI Text ----------
ui_text = {
    "English": {
        "title": "🌾 KrishiSahay",
        "caption": "AI Agricultural Field Assistant",
        "input": "Ask your farming question..."
    },
    "Hindi": {
        "title": "🌾 कृषि सहायक",
        "caption": "एआई कृषि सहायक",
        "input": "अपना कृषि प्रश्न पूछें..."
    },
    "Telugu": {
        "title": "🌾 కృషి సహాయ",
        "caption": "ఎఐ వ్యవసాయ సహాయకుడు",
        "input": "మీ వ్యవసాయ ప్రశ్న అడగండి..."
    }
}

language = st.selectbox(
    "Select Language / भाषा चुनें / భాషను ఎంచుకోండి",
    ["English", "Hindi", "Telugu"]
)

st.title(ui_text[language]["title"])
st.caption(ui_text[language]["caption"])

# ---------- Chat Memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input(ui_text[language]["input"])

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # Embed query
            query_embedding = embed_model.encode([query]).astype("float32")

            # Retrieve knowledge
            distances, indices = index.search(query_embedding, 3)
            retrieved_text = "\n\n".join([all_chunks[i] for i in indices[0]])

            prompt = f"""
You are KrishiSahay, a practical agricultural advisor.

Answer in {language}.
Keep response short (max 6 lines).
Use simple farmer-friendly language.

Use only this knowledge:
{retrieved_text}

Farmer Question:
{query}
"""

            answer = query_hf(prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )