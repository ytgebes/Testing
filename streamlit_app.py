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

# --- INITIALIZE SESSION STATE ---
# This holds the summary content to be displayed below the columns
if 'summary_content' not in st.session_state:
    st.session_state.summary_content = None
if 'summary_title' not in st.session_state:
    st.session_state.summary_title = None

# --- STYLING ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (This is the hamburger menu/sidebar) */
    [data-testid="stSidebar"] { display: none; }
    /* This also hides the auto-generated navigation menu */
    [data-testid="stPageLink"] { display: none; } 

    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }

    /* Custom Nav button container for the top-left */
    .nav-container {
        display: flex;
        justify-content: flex-start;
        padding-top: 2rem; /* PUSHES BUTTON DOWN */
        padding-bottom: 2rem;
    }
    .nav-button a {
        background-color: #7B1AF3; color: white; padding: 10px 20px;
        border-radius: 8px; text-decoration: none; font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .nav-button a:hover { background-color: #5F09C1; }

    /* Main Theme */
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
    .summary-card { /* New style for the full-width summary */
        background-color: #E6F0FF; 
        padding: 2rem; 
        border-radius: 10px; 
        margin-top: 2rem;
        border: 2px solid #6A1B9A;
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

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file_path): return pd.read_csv(file_path)

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    # ... (Keep this function exactly as it is) ...
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e: 
        return f"ERROR_FETCH: {e}"
    
    content_type = r.headers.get("Content-Type", "").lower()
    
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            with io.BytesIO(r.content) as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e: 
            return f"ERROR_PDF_PARSE: {e}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style']): tag.decompose()
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e: 
            return f"ERROR_HTML_PARSE: {e}"

def summarize_text_with_gemini(text: str):
    """Generates a summary using the Gemini API."""
    if not text or text.startswith("ERROR"): 
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"

    prompt = (f"Summarize this NASA bioscience paper. Output in clean Markdown with a section titled 'Key Findings' (using bullet points) and a section titled 'Overview Summary' (using a paragraph).\n\nContent:\n{text}")
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"ERROR_GEMINI: {e}"

# --- MAIN PAGE FUNCTION ---
def search_page():
    # --- NAVIGATION BUTTON (Custom HTML link) ---
    st.markdown(
        '<div class="nav-container"><div class="nav-button"><a href="/Assistant_AI" target="_self">Assistant AI 💬</a></div></div>',
        unsafe_allow_html=True
    )
    
    # --- UI Header ---
    df = load_data("SB_publication_PMC.csv")
    st.markdown('<h1>Simplified <span style="color: #6A1B9A;">Knowledge</span></h1>', unsafe_allow_html=True)
    st.markdown("### Search, Discover, and Summarize NASA's Bioscience Publications")

    search_query = st.text_input("Search publications...", placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")
    
    # --- Search Logic ---
    if search_query:
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(f"Found {len(results_df)} matching publications:")
        
        if results_df.empty:
            st.warning("No matching publications found.")
        else:
            # Clear previous summary content when a new search is run
            st.session_state.summary_content = None
            st.session_state.summary_title = None

            col_list = st.columns(2)
            col_idx = 0
            
            for idx, row in results_df.iterrows():
                current_col = col_list[col_idx % 2]
                with current_col:
                    with st.container():
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                        st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                        
                        # Use a function to set the state on button click
                        if st.button("🔬 Gather & Summarize", key=f"summarize_{idx}"):
                            # Set the title placeholder
                            st.session_state.summary_title = row['Title']
                            
                            # Run the summarization and store the result
                            with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                                text = fetch_url_text(row['Link'])
                                st.session_state.summary_content = summarize_text_with_gemini(text)
                            # Rerun the script to display the result outside the column
                            st.rerun() 
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                col_idx += 1
    
    # --- FULL-WIDTH SUMMARY DISPLAY ---
    # This check is performed outside the search_query block and outside the columns
    if st.session_state.summary_content:
        st.markdown("---")
        st.markdown(f'<div class="summary-card">', unsafe_allow_html=True)
        st.markdown(f"## 📄 Summary for: {st.session_state.summary_title}")
        st.markdown(st.session_state.summary_content)
        st.markdown("</div>", unsafe_allow_html=True)

# --- NAVIGATION SETUP ---

# This structure enables multi-page navigation correctly
pg = st.navigation([
    st.Page(search_page, title="Simplified Knowledge 🔍", icon="🏠"), # This is the main page function
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"), # This references the file in the pages folder
])

# Run the navigation
pg.run()
