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
    st.error("API Key not configured.")
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

# ---------- Language Dictionary ----------
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

# ---------- Language Selection ----------
language = st.selectbox(
    "Select Language / भाषा चुनें / భాషను ఎంచుకోండి",
    ["English", "Hindi", "Telugu"]
)

st.title(ui_text[language]["title"])
st.caption(ui_text[language]["caption"])

# ---------- Translation Function ----------
def translate_to_english(text):
    if language == "English":
        return text

    prompt = f"Translate the following into English:\n\n{text}"
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
query = st.chat_input(ui_text[language]["input"])

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # Step 1: Translate query to English internally
            internal_query = translate_to_english(query)

            # Step 2: Embed query
            query_embedding = embed_model.encode([internal_query]).astype("float32")

            # Step 3: Retrieve knowledge
            distances, indices = index.search(query_embedding, 3)
            retrieved_text = "\n\n".join([all_chunks[i] for i in indices[0]])

            # Step 4: Generate answer directly in selected language
            prompt = f"""
You are KrishiSahay, a practical agricultural advisor.

Answer in {language}.
Keep response short (max 6 lines).
Use simple farmer-friendly language.

Use only this knowledge:
{retrieved_text}

Farmer Question:
{internal_query}
"""

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            final_answer = response.text.strip()

            st.write(final_answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )