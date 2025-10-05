import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---

st.set_page_config(
    page_title="Simplified Knowledge",
    page_icon="🚀",
    layout="wide"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI. Please check your API key in Streamlit secrets. Details: {e}")
    st.stop()

# --- STYLING (NEW WHITE/LIGHT PURPLE THEME) ---

st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION BAR */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* NEW THEME: White Background */
    body {
        background-color: #FFFFFF;
        color: #333333; /* Dark grey for main text */
    }

    /* Titles and Headings */
    h1, h3 {
        color: #000000; /* Black for titles */
        text-align: center;
    }
    
    h1 {
        font-size: 4.5em !important;
        padding-bottom: 0.5rem;
    }

    /* Search Input Box */
    input[type="text"] {
        color: #000000 !important; /* Black text */
        background-color: #F0F2F6 !important; /* Light grey background */
        border: 1px solid #CCCCCC !important;
        border-radius: 8px;
        padding: 14px;
    }
    input::placeholder {
        color: #888888 !important; /* Lighter grey for placeholder */
    }

    /* Result Cards */
    .result-card {
        background-color: #FAFAFA; /* Very light grey for cards */
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Links */
    a {
        color: #6A1B9A; /* A darker purple for links for readability */
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* NEW: Light Purple Buttons */
    .stButton>button {
        border-radius: 8px;
        width: 100%;
        background-color: #E6E0FF; /* Light purple background */
        color: #4F2083; /* Darker purple text */
        border: 1px solid #C5B3FF;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D6C9FF;
        border: 1px solid #B098FF;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION LINK ---
st.markdown("[💬 Go to AI Chat Assistant](/1_🔬_AI_Chat_Assistant)", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    # (Helper function code is unchanged, kept for brevity)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NASA-App/1.0;)"}
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
    # (Helper function code is unchanged, kept for brevity)
    if not text or text.startswith("ERROR"): return text
    prompt = (f"Summarize this NASA paper. Output in Markdown with 'Key Findings' (bullets) and a 'Plain Language Summary' (paragraph).\n\nContent:\n{text[:25000]}")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        return model.generate_content(prompt).text
    except Exception as e: return f"ERROR_GEMINI: {e}"

# --- MAIN PAGE UI ---
df = load_data("SB_publication_PMC.csv")

st.title("Simplified Knowledge")
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
