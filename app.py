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
st.markdown("""
<style>

/* ===== Background ===== */
.stApp {
    background: linear-gradient(135deg, #f8fff4 0%, #e6f4ea 100%);
}

/* ===== Main Card ===== */
.block-container {
    background-color: white;
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 8px 25px rgba(0, 100, 0, 0.08);
}

/* ===== GLOBAL FONT SIZE INCREASE ===== */
html, body, [class*="css"] {
    font-size: 18px !important;
    color: #000000 !important;
}

/* Headings Bigger */
h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    color: #1b5e20 !important;
}

h2, h3 {
    font-size: 26px !important;
    color: #1b5e20 !important;
}

/* Caption */
p {
    font-size: 18px !important;
    color: #000000 !important;
}

/* ===== TOP GREEN INFO TEXT FIX ===== */
div[style*="border-left"] {
    color: #000000 !important;
    font-size: 19px !important;
}

/* ===== Chat Bubbles ===== */
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 14px 18px;
    background-color: #e8f5e9;
    border: 1px solid #a5d6a7;
    font-size: 18px !important;
}
            /* Ensure paragraphs inside chat are black */
[data-testid="stChatMessage"] p {
    color: #000000 !important;
}

/* Ensure list text black */
[data-testid="stChatMessage"] li {
    color: #000000 !important;
}

/* ===== Input Area ===== */
textarea {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    font-size: 18px !important;
    border-radius: 12px !important;
    border: 1px solid #66bb6a !important;
}

/* Placeholder color */
textarea::placeholder {
    color: #cccccc !important;
    font-size: 17px !important;
}

/* Send button */
button[kind="secondary"] {
    background-color: #2e7d32 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Select box */
.stSelectbox > div > div {
    background-color: #f1f8e9;
    border-radius: 10px;
    border: 1px solid #66bb6a;
    font-size: 17px !important;
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)
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

# ---------- Multilingual Configuration ----------
# Define UI text and instructions for each language
LANGUAGES = {
    "English": {
        "title": "KrishiSahay 🌾",
        "caption": "AI Agricultural Field Assistant",
        "placeholder": "Ask your farming question...",
        "instruction": "Answer strictly in English.",
        "select_lang": "Select Language",
        "scheme_header": "Government Scheme Advisor",
        "advice_header": "Agricultural Field Officer"
    },
    "हिंदी (Hindi)": {
        "title": "कृषिसहाय 🌾",
        "caption": "कृषि क्षेत्र सहायक AI",
        "placeholder": "अपनी खेती से संबंधित प्रश्न पूछें...",
        "instruction": "कृपया हिंदी में उत्तर दें।",
        "select_lang": "भाषा चुनें",
        "scheme_header": "सरकारी योजना सलाहकार",
        "advice_header": "कृषि क्षेत्र अधिकारी"
    },
    "తెలుగు (Telugu)": {
        "title": "కృషిసహాయం 🌾",
        "caption": "AI వ్యావసాయिक ఫీల్డ్ ఎసిస్టెంట్",
        "placeholder": "మీ వ్యావసాయ ప్రశ्नలు అడుగుపై...",
        "instruction": "దయచేసి తెలుగులో సమాధానం ఇవ్వండి.",
        "select_lang": "భాష ఎంచుకోండి",
        "scheme_header": "Government Scheme Advisor",
        "advice_header": "Agricultural Field Officer"
    }
}

# Crop translations for search filtering
CROP_TRANSLATIONS = {
    "wheat": ["wheat", "गेहूं", "గోధుమ"],
    "rice": ["rice", "चावल", "బియ్యం"],
    "cotton": ["cotton", "कपास", "పత్తి"],
    "sugarcane": ["sugarcane", "गन्ना", "చెరుకు"],
    "paddy": ["paddy", "धान", " వరి"],
    "maize": ["maize", "मक्का", "మ్కी"],
    "groundnut": ["groundnut", "मूंगफली", " బఠానी"]
}

# ---------- Load Embedding Model ----------
# Supports 50+ languages including Hindi and Telugu
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

# ---------- Crop Detection (Multilingual) ----------
def detect_crop(query):
    query_lower = query.lower()
    crops = list(set(all_categories))
    
    # Check English keywords
    for crop in crops:
        if crop in query_lower:
            return crop
            
    # Check translations
    for eng_crop, translations in CROP_TRANSLATIONS.items():
        if eng_crop in all_categories:
            if any(t in query_lower for t in translations):
                return eng_crop
                
    return None

# ---------- UI ----------
# Language Selector at the top
col1, col2 = st.columns([1, 3])
with col1:
    selected_lang = st.selectbox(
        "Language / భाषा / भाषा",
        options=list(LANGUAGES.keys()),
        index=0
    )

# Get current language settings
lang = LANGUAGES[selected_lang]

# Apply dynamic Title and Caption
st.title(lang["title"])
st.caption(lang["caption"])

# Reset chat if language changes
if "current_lang" not in st.session_state:
    st.session_state.current_lang = selected_lang
elif st.session_state.current_lang != selected_lang:
    st.session_state.messages = []
    st.session_state.current_lang = selected_lang
    st.rerun()

# Display Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Dynamic Input Placeholder
query = st.chat_input(lang["placeholder"])

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🌾"):

            # -------- Embed Query (Multilingual) --------
            query_embedding = embed_model.encode([query]).astype("float32")

            # -------- Crop Detection --------
            detected_crop = detect_crop(query)

            # -------- Retrieval --------
            distances, indices = index.search(query_embedding, 7)

            retrieved_chunks = []

            # Filter by crop if detected
            for i in indices[0]:
                if detected_crop:
                    if all_categories[i] == detected_crop:
                        retrieved_chunks.append(all_chunks[i])
                else:
                    retrieved_chunks.append(all_chunks[i])

            # Fallback if not enough chunks found
            if len(retrieved_chunks) < 2:
                retrieved_chunks = [all_chunks[i] for i in indices[0]]

            retrieved_text = "\n\n".join(retrieved_chunks[:4])

            # -------- Scheme Intent Detection --------
            # Keywords in English, Hindi, and Telugu
            scheme_keywords = [
                "scheme", "subsidy", "loan", "pm kisan", "insurance", 
                "benefit", "eligibility", "government", "yojana",
                "योजना", " subsidy", "malinya", "vaddhu", "vimanam"
            ]

            is_scheme_query = any(word in query.lower() for word in scheme_keywords)

            # -------- Dynamic Prompt --------
            lang_instruction = lang["instruction"]
            header = lang["scheme_header"] if is_scheme_query else lang["advice_header"]

            if is_scheme_query:
                prompt = f"""
You are an agricultural government scheme advisor ({header}).

{lang_instruction}

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
You are an experienced agricultural field officer ({header}) helping farmers.

{lang_instruction}

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