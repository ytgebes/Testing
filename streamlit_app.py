import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Simplified Knowledge", layout="wide")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION */
    [data-testid="stSidebar"] { display: none; }

    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }

    /* UPDATED: Nav button container aligned to the left */
    .nav-container {
        display: flex;
        justify-content: flex-start; /* Aligns button to the left */
        padding-bottom: 2rem; /* Adds space below the button */
    }
    .nav-button a {
        background-color: #7B1AF3; color: white; padding: 10px 20px;
        border-radius: 8px; text-decoration: none; font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .nav-button a:hover { background-color: #5F09C1; }

    /* Main Theme */
    body { background-color: #FFFFFF; color: #333333; }
    h1, h3 { text-align: center; }
    h1 { font-size: 4.5em !important; padding-bottom: 0.5rem; color: #000000; }
    h3 { color: #333333; }
    input[type="text"] {
        color: #000000 !important; background-color: #F0F2F6 !important;
        border: 1px solid #CCCCCC !important; border-radius: 8px; padding: 14px;
    }
    .result-card {
        background-color: #FAFAFA; padding: 1.5rem; border-radius: 10px;
        margin-bottom: 1rem; border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }
    .stButton>button {
        border-radius: 8px; width: 100%; background-color: #E6E0FF;
        color: #4F2083; border: 1px solid #C5B3FF; font-weight: bold;
    }
    .stButton>button:hover { background-color: #D6C9FF; border: 1px solid #B098FF; }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION BUTTON (MOVED TO TOP-LEFT) ---
st.markdown(
    # UPDATED: Link points to the new page name
    '<div class="nav-container"><div class="nav-button"><a href="/Assistant_AI" target="_self">Assistant AI 💬</a></div></div>',
    unsafe_allow_html=True
)

# --- HELPER FUNCTIONS (UNCHANGED) ---
@st.cache_data
def load_data(file_path): return pd.read_csv(file_path)

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e: return f"ERROR_FETCH: {e}"
    content_type = r.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            with io.BytesIO(r.content) as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e: return f"ERROR_PDF_PARSE: {e}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style']): tag.decompose()
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e: return f"ERROR_HTML_PARSE: {e}"

def summarize_text_with_gemini(text: str):
    if not text or text.startswith("ERROR"): return text
    prompt = (f"Summarize this NASA paper. Output in Markdown with 'Key Findings' (bullets) and a 'Plain Language Summary' (paragraph).\n\nContent:\n{text[:25000]}")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        return model.generate_content(prompt).text
    except Exception as e: return f"ERROR_GEMINI: {e}"

# --- MAIN PAGE UI ---
df = load_data("SB_publication_PMC.csv")

# UPDATED: Title with custom purple color on the second word
st.markdown('<h1>Simplified <span style="color: #6A1B9A;">Knowledge</span></h1>', unsafe_allow_html=True)
st.markdown("### Search, Discover, and Summarize NASA's Bioscience Publications")

search_query = st.text_input("Search publications...", placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")

if search_query:
    mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
    results_df = df[mask].reset_index(drop=True)
    st.markdown("---")
    st.subheader(f"Found {len(results_df)} matching publications:")
    if results_df.empty:
        st.warning("No matching publications found.")
    else:
        for idx, row in results_df.iterrows():
            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<h4><a href='{row['Link']}' target='_blank'>{row['Title']}</a></h4>", unsafe_allow_html=True)
                if st.button("🔬 Gather & Summarize", key=f"summarize_{idx}"):
                    placeholder = st.empty()
                    with st.spinner("Accessing and summarizing content..."):
                        text = fetch_url_text(row['Link'])
                        summary = summarize_text_with_gemini(text)
                        placeholder.markdown(summary)
                st.markdown("</div>", unsafe_allow_html=True)
