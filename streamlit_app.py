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
if 'summary_content' not in st.session_state:
    st.session_state.summary_content = None
if 'summary_title' not in st.session_state:
    st.session_state.summary_title = None

# --- STYLING ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (This is the sidebar hamburger menu) */
    [data-testid="stSidebar"] { display: none; }
    /* HIDE THE AUTO-GENERATED LINKS FROM st.navigation */
    [data-testid="stPageLink"] { display: none; } 

    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }

    /* REMOVE custom Nav button styling as st.navigation handles it now */
    .nav-container { display: none; }

    /* Main Theme */
    h1, h3 { text-align: center; }
    h1 { font-size: 4.5em !important; padding-bottom: 0.5rem; color: #000000; }
    h3 { color: #333333; }
    input[type="text"] {
        color: #000000 !important; background-color: #F0F2F6 !important;
        border: 1px solid #CCCCCC !important; border-radius: 8px; padding: 14px;
    }
    /* RESULT CARD: Remains the same, but now takes full width */
    .result-card {
        background-color: #FAFAFA; padding: 1.5rem; border-radius: 10px;
        margin-bottom: 1rem; border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .summary-card { 
        background-color: #E6F0FF; 
        padding: 2rem; 
        border-radius: 10px; 
        margin-top: 2rem;
        border: 2px solid #6A1B9A;
    }
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }
    /* BUTTON: Now uses a smaller width since it's in a single column */
    .stButton>button {
        border-radius: 8px; 
        width: 50%; /* Adjusting width for better look in single column */
        min-width: 250px; /* Ensure it's big enough */
        background-color: #E6E0FF;
        color: #4F2083; border: 1px solid #C5B3FF; font-weight: bold;
    }
    .stButton>button:hover { background-color: #D6C9FF; border: 1px solid #B098FF; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS (UNCHANGED) ---
@st.cache_data
def load_data(file_path): 
    """Loads the publication data."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure 'SB_publication_PMC.csv' is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    """Fetches text content from a URL, handling HTML and basic PDF parsing."""
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
    # --- NAVIGATION LINK ---
    # Using st.page_link for official navigation and better UX placement
    st.page_link("pages/Assistant_AI.py", label="Assistant AI 💬", icon="💬")
    
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
            
            # 🟢 SINGLE COLUMN DISPLAY LOOP
            for idx, row in results_df.iterrows():
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    # Center the button in the single-column view
                    col_spacer, col_btn, col_spacer_2 = st.columns([1, 2, 1])
                    with col_btn:
                        if st.button("🔬 Gather & Summarize", key=f"summarize_{idx}"):
                            
                            st.session_state.summary_title = row['Title']
                            
                            with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                                try:
                                    text = fetch_url_text(row['Link'])
                                    summary = summarize_text_with_gemini(text)
                                    st.session_state.summary_content = summary
                                except Exception as e:
                                    st.session_state.summary_content = f"**Critical Error during summarization:** {e}. Please check your API key and the publication link."

                            # Rerun to display the summary outside the column
                            st.rerun() 
                            
                    st.markdown("</div>", unsafe_allow_html=True)
                    # Add space between cards
                    st.write("") 
    
    # --- FULL-WIDTH SUMMARY DISPLAY ---
    if st.session_state.summary_content:
        st.markdown("---")
        st.markdown(f'<div class="summary-card">', unsafe_allow_html=True)
        
        title = st.session_state.summary_title
        if "Critical Error" in st.session_state.summary_content:
             st.markdown(f"## ❌ Failed to Summarize: {title}")
        else:
             st.markdown(f"## 📄 Summary for: {title}")
             
        st.markdown(st.session_state.summary_content)
        st.markdown("</div>", unsafe_allow_html=True)

# --- NAVIGATION SETUP ---
pg = st.navigation([
    st.Page(search_page, title="Simplified Knowledge 🔍", icon="🏠"),
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"),
])

pg.run()
