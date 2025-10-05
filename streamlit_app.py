import streamlit as st
import pandas as pd
import io
import time
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai
from streamlit_extras.let_it_rain import rain

# --- CONFIGURATION & INITIALIZATION ---

st.set_page_config(
    page_title="NASA Simplified Knowledge",
    page_icon="🚀",
    layout="wide"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-1.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI. Please check your API key in Streamlit secrets. Details: {e}")
    st.stop()

# --- STYLING (MODIFIED) ---

st.markdown("""
    <style>
    /* MODIFIED: Reduce top padding to push content up */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* MODIFIED: New purple and white theme */
    body {
        background-color: #1e1032; /* Deep purple background */
        color: white;
    }
    
    /* MODIFIED: Title styling */
    h1 {
        font-size: 4.5em !important; /* Even larger title */
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
        padding-bottom: 1rem;
    }
    
    /* MODIFIED: Subheading styling */
    h3 {
        text-align: center;
        color: #d8b8ff !important; /* Light lavender for subheading */
    }

    /* MODIFIED: Search input styling */
    input[type="text"] {
        color: white !important;
        background-color: #3a215b !important; /* Lighter purple input box */
        border: 1px solid #4f2083 !important;
        border-radius: 8px;
        padding: 14px;
    }
    input::placeholder {
        color: #d8b8ff !important;
    }

    /* MODIFIED: Custom result card styling */
    .result-card {
        background-color: #2c1a47; /* Contrasting purple for cards */
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #4f2083;
        transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: scale(1.01);
        border-color: #d8b8ff; /* Lavender border on hover */
    }
    
    /* MODIFIED: Link styling */
    a {
        color: #FFFFFF; /* Bright white for links for high contrast */
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* Button styling for a cohesive look */
    .stButton>button {
        border-radius: 8px;
        width: 100%;
        background-color: #4f2083;
        color: white;
        border: 1px solid #d8b8ff;
    }
    .stButton>button:hover {
        background-color: #d8b8ff;
        color: #1e1032;
        border: 1px solid #d8b8ff;
    }
    </style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS (CACHED) ---

@st.cache_data
def load_data(file_path):
    """Loads the NASA publications CSV into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: The data file '{file_path}' was not found. Please make sure it's in the correct directory.")
        st.stop()

@lru_cache(maxsize=128)
def fetch_url_text(url: str) -> str:
    """Download a URL and return its text content (handles PDF and HTML)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NASA-App/1.0; +http://your-app-url.com)"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"ERROR_FETCH: Could not retrieve content from URL. Reason: {str(e)}"

    content_type = r.headers.get("Content-Type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            with io.BytesIO(r.content) as pdf_bytes:
                reader = PyPDF2.PdfReader(pdf_bytes)
                if not reader.pages:
                    return "ERROR_EXTRACT: PDF is empty or corrupted."
                text_parts = [p.extract_text() for p in reader.pages if p.extract_text()]
                return "\n".join(text_parts) if text_parts else "ERROR_EXTRACT: No text could be extracted from this PDF."
        except Exception as e:
            return f"ERROR_PDF_PARSE: Failed to parse the PDF file. Reason: {str(e)}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            body = soup.body
            if body:
                return " ".join(body.get_text(separator=" ", strip=True).split())[:25000]
            return "ERROR_EXTRACT: Could not find the main body content of the webpage."
        except Exception as e:
            return f"ERROR_HTML_PARSE: Failed to parse the HTML content. Reason: {str(e)}"

def summarize_text_with_gemini(text: str) -> str:
    """Generates a summary of the provided text using the Gemini model."""
    if not text or text.startswith("ERROR"):
        return text

    context = text[:25000]
    prompt = (
        "You are an expert scientific analyst. Summarize the key findings from the following NASA bioscience publication content. "
        "Provide your output in Markdown format with two sections:\n\n"
        "1.  **Key Findings (3-4 bullet points):** List the most critical discoveries or conclusions.\n"
        "2.  **Plain Language Summary:** Provide a concise, easy-to-understand paragraph (2-3 sentences) explaining the research and its significance.\n\n"
        f"**Publication Content:**\n---\n{context}"
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR_GEMINI: The AI summary could not be generated. Reason: {str(e)}"

# --- MAIN PAGE UI ---

df = load_data("SB_publication_PMC.csv")

st.title("Simplified Knowledge")
st.markdown("### Search, Discover, and Summarize NASA's Bioscience Publications")

search_query = st.text_input(
    "Enter a keyword to search all 608 NASA publications...",
    placeholder="e.g., microgravity, radiation, Artemis, cell biology...",
    key="search_box",
    label_visibility="collapsed" # Hides the label for a cleaner look
)

if search_query:
    mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
    results_df = df[mask].reset_index(drop=True)

    st.markdown("---")
    st.subheader(f"Found {len(results_df)} matching publications:")

    if results_df.empty:
        st.warning("No matching publications found. Please try a different keyword.")
    else:
        for idx, row in results_df.iterrows():
            title = row["Title"]
            link = row["Link"]

            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<h4><a href='{link}' target='_blank'>{title}</a></h4>", unsafe_allow_html=True)

                if st.button("🔬 Gather & Summarize", key=f"summarize_{idx}"):
                    summary_placeholder = st.empty()
                    with st.spinner("Accessing publication content... This may take a moment."):
                        extracted_text = fetch_url_text(link)

                    if extracted_text.startswith("ERROR"):
                        summary_placeholder.error(extracted_text)
                    else:
                        summary_placeholder.info("Content retrieved successfully. Generating AI summary...")
                        with st.spinner("🤖 Gemini AI is reading and summarizing the paper..."):
                            summary = summarize_text_with_gemini(extracted_text)
                            summary_placeholder.markdown(summary)

                st.markdown("</div>", unsafe_allow_html=True)

# REMOVED: Deleted the 'else' block and 'st.info' message for a cleaner initial view.
