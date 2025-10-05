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

# Configure the page
st.set_page_config(
    page_title="NASA Simplified Knowledge",
    page_icon="🚀",
    layout="wide"
)

# Configure Gemini AI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-1.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI. Please check your API key in Streamlit secrets. Details: {e}")
    st.stop()

# --- STYLING ---

st.markdown("""
    <style>
    /* Main background and text color */
    body {
        background-color: #0b3d91;
        color: white;
    }
    /* Page title styling */
    .st-emotion-cache-10trblm {
        font-size: 4em; /* Much larger title */
        font-weight: bold;
        text-align: center;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    /* Search input styling */
    input[type="text"] {
        color: white !important;
        background-color: #1e1e2f !important;
        border: 1px solid #444 !important;
        border-radius: 8px;
        padding: 12px;
    }
    input::placeholder {
        color: #cccccc !important;
    }
    /* Custom result card styling */
    .result-card {
        background-color: #0e2a6b;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #1c4b82;
        transition: transform 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: scale(1.02);
        border: 1px solid #00ffcc;
    }
    a {
        color: #00ffcc;
        text-decoration: none;
    }
    .stButton>button {
        border-radius: 8px;
        width: 100%;
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

    # Handle PDF content
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
    # Handle HTML content
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

    # Truncate text to stay within model limits and improve performance
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

# Load the dataset
df = load_data("SB_publication_PMC.csv")

st.title("Simplified Knowledge")
st.markdown("<h3 style='text-align: center; color: #cccccc;'>Search, Discover, and Summarize NASA's Bioscience Publications</h3>", unsafe_allow_html=True)

# --- SEARCH BAR ---
search_query = st.text_input(
    "Enter a keyword to search all 608 NASA publications...",
    placeholder="e.g., microgravity, radiation, Artemis, cell biology...",
    key="search_box"
)

# --- DISPLAY RESULTS ---
if search_query:
    # Perform a case-insensitive search on the 'Title' column
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

                # Use a unique key for each button by combining prefix and index
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

else:
    st.info("The search bar is ready. Start typing to find relevant NASA research.")
