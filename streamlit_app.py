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
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (This is the sidebar hamburger menu) */
    [data-testid="stSidebar"] { display: none; }
    
    /* 🟢 FIX 1: REMOVE CSS HIDING st.page_link */
    /* [data-testid="stPageLink"] { display: none; } */ 
    /* The link will now show */

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
        margin-bottom: 1rem; 
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Increase Title Font Size */
    .result-card a { 
        font-size: 1.15em; 
        display: block; 
        margin-bottom: 10px; 
    }

    /* Summary Card Container (The outer container that is NOT the summary-display) */
    .summary-card { 
        background-color: #E6F0FF; 
        padding: 2rem; 
        border-radius: 10px; 
        margin-top: 2rem;
        border: 2px solid #6A1B9A;
    }
    
    /* Consistent Purple Link Color */
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }
    
    /* BUTTON: Now uses a smaller width in single column and aligns left */
    .stButton>button {
        border-radius: 8px; 
        width: auto; 
        min-width: 200px; 
        background-color: #E6E0FF;
        color: #4F2083; border: 1px solid #C5B3FF; font-weight: bold;
        display: block; 
    }
    .stButton>button:hover { background-color: #D6C9FF; border: 1px solid #B098FF; }
    
    /* 🟢 FIX 2: Remove styling for the inner summary box (summary-display) */
    .summary-display {
        /* background-color: #FFF; */ /* REMOVED */
        padding: 0; /* REMOVED */
        /* border-radius: 8px; */ /* REMOVED */
        border: none; /* REMOVED */
        margin-top: 1rem;
    }
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
    # This link should now be visible due to the CSS fix.
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
            if 'summary_dict' not in st.session_state:
                 st.session_state.summary_dict = {}
            
            # SINGLE COLUMN DISPLAY LOOP
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Display title first with custom styling
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    # Place button immediately below title
                    if st.button("🔬 Gather & Summarize", key=f"btn_summarize_{idx}"):
                        
                        # GENERATE SUMMARY IMMEDIATELY UPON CLICK
                        with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                            try:
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"**Critical Error during summarization:** {e}. Please check the link and API key."
                        
                        # We use st.rerun() here to force the display logic (below) to run 
                        # after the long operation has updated the session state.
                        st.rerun()

                    # DISPLAY SUMMARY IF IT EXISTS FOR THIS PUBLICATION
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        # 🟢 FIX 2: Use summary-display class, but the CSS is now neutral
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        st.markdown(f"---", unsafe_allow_html=True) # Separator for cleaner look
                        
                        if "Critical Error" in summary_content or summary_content.startswith("ERROR"):
                            st.markdown(f"**❌ Failed to Summarize:** *{row['Title']}*", unsafe_allow_html=True)
                            st.markdown(f"**Error Details:** {summary_content}")
                        else:
                            st.markdown(f"**📄 Summary for:** *{row['Title']}*", unsafe_allow_html=True)
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True)
                    # Add space between cards
                    st.write("") 
    

# --- NAVIGATION SETUP ---
pg = st.navigation([
    st.Page(search_page, title="Simplified Knowledge 🔍", icon="🏠"),
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"),
])

pg.run()
