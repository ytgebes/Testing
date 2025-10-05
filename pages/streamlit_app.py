import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai

# --- CONFIGURATION & GLOBAL CONSTANTS ---
st.set_page_config(page_title="Simplified Knowledge", layout="wide")

try:
    # Configure the Gemini API using Streamlit's secrets management
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.5-flash"
    DATA_FILE = "SB_publication_PMC.csv"
except Exception as e:
    st.error(f"Error initializing Application: {e}")
    # Do not stop, allow the rest of the app to render the error
    # st.stop() # Removed st.stop() for stability

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""


# --- STYLING (Main Page CSS) ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (Sidebar hamburger menu is now the main navigation) */
    /* DO NOT HIDE [data-testid="stSidebar"] or you will lose multi-page navigation! */

    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }

    /* Main Theme */
    h1, h3 { text-align: center; }
    h1 { font-size: 4.5em !important; padding-bottom: 0.5rem; color: #000000; }
    h3 { color: #333333; }
    input[type="text"] {
        color: #000000 !important; background-color: #F0F2F6 !important;
        border: 1px solid #CCCCCC !important; border-radius: 8px; padding: 14px;
    }
    
    /* Result Card Styling */
    .result-card {
        background-color: #FAFAFA; 
        padding: 1.5rem; 
        border-radius: 10px;
        margin-bottom: 1.5rem; 
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-card .stMarkdown strong { 
        font-size: 1.15em; 
        display: block;
        margin-bottom: 10px; 
    }

    /* Links and Buttons */
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }
    
    .stButton>button {
        border-radius: 8px; 
        width: auto; 
        min-width: 200px; 
        background-color: #E6E0FF;
        color: #4F2083; 
        border: 1px solid #C5B3FF; 
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #D6C9FF; border: 1px solid #B098FF; }

    /* Summary Display Styling */
    .summary-display {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed #CCC;
    }
    .summary-display h3 {
        text-align: left !important;
        color: #4F2083;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 1.3em;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS: DATA, FETCHING, & AI ---

@st.cache_data
def load_data(file_path): 
    """Loads the publication data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure '{file_path}' is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    """Fetches and extracts text content from a URL (HTML or PDF)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e: 
        return f"ERROR_FETCH: Request failed - {e}"
    
    content_type = r.headers.get("Content-Type", "").lower()
    
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            with io.BytesIO(r.content) as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e: 
            return f"ERROR_PDF_PARSE: PDF reading failed - {e}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style']): tag.decompose()
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e: 
            return f"ERROR_HTML_PARSE: HTML parsing failed - {e}"

def summarize_text_with_gemini(text: str):
    """Generates a structured summary using the Gemini API."""
    if not text or text.startswith("ERROR"): 
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"

    prompt = (
        f"Summarize this NASA bioscience paper. Output in clean Markdown with a "
        f"level 3 heading (###) titled 'Key Findings' (using bullet points) and "
        f"a level 3 heading (###) titled 'Overview Summary' (using a paragraph).\n\nContent:\n{text}"
    )
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"ERROR_GEMINI: API call failed - {e}"

# --- MAIN PAGE FUNCTION (RENAMED TO HOME) ---

def run_app_page():
    """Renders the main search, results, and summarization interface."""
        
    # --- UI Header ---
    df = load_data(DATA_FILE)
    st.markdown('<h1>Simplified <span style="color: #6A1B9A;">Knowledge</span></h1>', unsafe_allow_html=True)
    st.markdown("### Search, Discover, and Summarize NASA's Bioscience Publications")

    # Search Input
    search_query = st.text_input("Search publications...", placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")
    
    # --- Search & Results Logic ---
    if search_query:
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(f"Found {len(results_df)} matching publications:")
        
        if results_df.empty:
            st.warning("No matching publications found.")
        else:
            
            if st.session_state.last_query != search_query:
                st.session_state.summary_dict = {}
                st.session_state.last_query = search_query

            # Loop through results and create a card for each
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Title (linked to source)
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    # Summarize Button
                    if st.button("🔬 Gather & Summarize", key=f"btn_summarize_{idx}"):
                        
                        # Execute summarization when button is pressed
                        with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                            try:
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: Unexpected error during summary: {e}"
                        
                        st.rerun()

                    # Display Summary if it exists in the session state
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f"**❌ Failed to Summarize:** *{row['Title']}*", unsafe_allow_html=True)
                            st.error(f"Error details: {summary_content}")
                        else:
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True) 

# --- APPLICATION ENTRY POINT ---
# Streamlit runs this function automatically because the file is in the pages/ directory
run_app_page()
