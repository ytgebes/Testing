import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai
import json # Import json for translation functions
from streamlit_extras.let_it_rain import rain # Assuming this is required for the effect

# --- CONFIGURATION ---
st.set_page_config(page_title="Simplified Knowledge", layout="wide")

try:
    # Use st.secrets for security in deployment
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.5-flash"
    DATA_FILE = "SB_publication_PMC.csv"
    
    # Placeholder for UI strings (English defaults)
    UI_STRINGS_EN = {
        "title": "Simplified Knowledge",
        "description": "Search, Discover, and Summarize NASA's Bioscience Publications",
        "search_placeholder": "e.g., microgravity, radiation, Artemis...",
        "button_summarize": "🔬 Gather & Summarize",
        "search_header": "matching publications:",
        "warning_empty": "No matching publications found."
    }
except Exception as e:
    st.error(f"Error configuring AI or defining constants: {e}")
    st.stop()

# --- LANGUAGES DICTIONARY (REQUIRED FOR SELECTBOX) ---
LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "Русский": {"label": "Русский (Russian)", "code": "ru"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
    "한국어": {"label": "한국어 (Korean)", "code": "ko"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
    "العربية": {"label": "العربية (Arabic)", "code": "ar"},
    # ... (Include all your other languages here) ...
}

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    # Initialize translations with English defaults
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()} 
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""


# --- STYLING (Main Page) ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT SIDEBAR (including the multi-page menu) */
    [data-testid="stSidebar"] { display: none; }
    
    /* NEW: Custom Nav buttons container for the top of the page */
    .nav-container-ai {
        display: flex;
        justify-content: space-between; /* Puts buttons on the left, dropdown on the right */
        align-items: center;
        padding-top: 3rem; 
        padding-bottom: 0rem;
        margin-bottom: 1rem;
    }
    
    /* Container for the two custom buttons (left side) */
    .nav-buttons-left {
        display: flex;
        gap: 15px; /* Space between the two custom buttons */
    }

    .nav-button-style a {
        background-color: #6A1B9A; /* Purple color */
        color: white !important; 
        padding: 10px 20px;
        border-radius: 8px; 
        text-decoration: none; 
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .nav-button-style a:hover { 
        background-color: #4F0A7B; /* Darker purple on hover */
    }
    
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

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path): 
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure 'SB_publication_PMC.csv' is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
        
def extract_json_from_text(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end+1])

def translate_dict_via_gemini(source_dict: dict, target_lang_name: str):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        f"Translate the VALUES of the following JSON object into {target_lang_name}.\n"
        "Return ONLY a JSON object with the same keys and translated values (no commentary).\n"
        f"Input JSON:\n{json.dumps(source_dict, ensure_ascii=False)}\n"
    )
    resp = model.generate_content(prompt)
    return extract_json_from_text(resp.text)

def translate_list_via_gemini(items: list, target_lang_name: str):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        f"Translate this list of short strings into {target_lang_name}. "
        f"Return a JSON array of translated strings in the same order.\n"
        f"Input: {json.dumps(items, ensure_ascii=False)}\n"
    )
    resp = model.generate_content(prompt)
    start = resp.text.find('[')
    end = resp.text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output.")
    return json.loads(resp.text[start:end+1])

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
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
    if not text or text.startswith("ERROR"): 
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"

    prompt = (f"Summarize this NASA bioscience paper. Output in clean Markdown with a level 3 heading (###) titled 'Key Findings' (using bullet points) and a level 3 heading (###) titled 'Overview Summary' (using a paragraph).\n\nContent:\n{text}")
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"ERROR_GEMINI: {e}"


# --- MAIN PAGE FUNCTION ---
def search_page():
    
    # --- TOP NAVIGATION BAR (HTML Buttons and Language Dropdown) ---
    # This block replaces the need for st.navigation and st.sidebar
    
    # 1. Custom HTML Links (Left Side)
    st.markdown("""
        <div class="nav-container-ai">
            <div class="nav-buttons-left">
                <div class="nav-button-style">
                    <a href="/Assistant_AI" target="_self">Assistant AI 💬</a>
                </div>
                <div class="nav-button-style">
                    <a href="/Translator" target="_self">Language Translator 🌎</a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Language Selection (Right Side)
    # Use st.columns to push the selectbox to the far right.
    lang_spacer_col, lang_select_col = st.columns([5, 1.5]) 

    with lang_select_col:
        lang_choice = st.selectbox(
            "🌐 Choose language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x]["label"],
            index=list(LANGUAGES.keys()).index(st.session_state.current_lang),
            label_visibility="collapsed" # Hide the label for a cleaner look
        )

    # --- Translation Logic (Must run immediately after selection) ---
    global translated_strings
    if lang_choice != st.session_state.current_lang:
        rain(emoji="⏳", font_size=54, falling_speed=5, animation_length=2)
        with st.spinner(f"Translating UI to {lang_choice}..."):
            try:
                if lang_choice in st.session_state.translations:
                    translated_strings = st.session_state.translations[lang_choice]
                else:
                    translated_strings = translate_dict_via_gemini(
                        st.session_state.translations["English"],
                        lang_choice
                    )
                    st.session_state.translations[lang_choice] = translated_strings
                st.session_state.current_lang = lang_choice
            except Exception as e:
                st.error("Translation failed — using English. Error: " + str(e))
                translated_strings = st.session_state.translations["English"]
                st.session_state.current_lang = "English"
        st.rerun() # Rerun to apply translation strings immediately
    else:
        # Load current translated strings
        translated_strings = st.session_state.translations[st.session_state.current_lang]
        
    # --- UI Header (Using translated strings) ---
    st.markdown(f"<h1>{translated_strings.get('title', 'Simplified Knowledge')}</h1>", unsafe_allow_html=True)
    
    # *** FIXED LINE: Using double quotes for the default string to avoid f-string SyntaxError ***
    st.markdown(f"### {translated_strings.get('description', "Search, Discover, and Summarize NASA's Bioscience Publications")}")
    
    search_query = st.text_input(
        "Search publications...", 
        placeholder=translated_strings.get('search_placeholder', "e.g., microgravity, radiation, Artemis..."), 
        label_visibility="collapsed"
    )
    
    # --- Search Logic ---
    df = load_data(DATA_FILE)
    if search_query:
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        
        # Use translated string for header
        st.subheader(f"Found {len(results_df)} {translated_strings.get('search_header', 'matching publications:')}")
        
        if results_df.empty:
            st.warning(translated_strings.get('warning_empty', "No matching publications found."))
        else:
            if st.session_state.get('last_query') != search_query:
                st.session_state.summary_dict = {}
                st.session_state.last_query = search_query

            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    # Use translated string for button
                    if st.button(translated_strings.get('button_summarize', "🔬 Gather & Summarize"), key=f"btn_summarize_{idx}"):
                        with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                            try:
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        st.rerun()

                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f"**❌ Failed to Summarize:** *{row['Title']}*", unsafe_allow_html=True)
                            st.error(f"Error fetching/summarizing content: {summary_content}")
                        else:
                            st.markdown(summary_content)
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True) 
    

# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    search_page()
