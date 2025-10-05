import streamlit as st
import json
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
from streamlit_extras.let_it_rain import rain
import google.generativeai as genai

# --------------------------- CONFIG ---------------------------
MODEL_NAME = "gemini-2.5-flash"
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel(MODEL_NAME)

# --------------------------- LOAD DATA ---------------------------
df = pd.read_csv("SB_publication_PMC.csv")  # CSV with "Title" and "Link"

# --------------------------- GLOBAL STYLING ---------------------------
st.set_page_config(page_title="Simplified Knowledge", layout="wide")
st.markdown("""
<style>
body { background-color: #0b3d91; color: white; }
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    margin-top: -20px;
    margin-bottom: 10px;
    color: white;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cccccc;
    margin-bottom: 40px;
}
.result-card {
    background-color: #0e2a6b;
    padding: 12px;
    border-radius:8px;
    margin-bottom:10px;
}
input[type="text"] {
    color: white !important;
    background-color: #1e1e2f !important;
    border: 1px solid #444 !important;
    font-size: 18px !important;
    text-align: center;
}
input::placeholder {
    color: #cccccc !important;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- HELPER FUNCTIONS ---------------------------
def extract_json_from_text(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end+1])

def translate_list_via_gemini(items: list, target_lang_name: str):
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

@lru_cache(maxsize=256)
def fetch_url_text(url: str) -> str:
    """Download url and return extracted text (PDF or HTML). Cached in-memory."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NASA-App/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return f"ERROR_FETCH: {str(e)}"

    content_type = r.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            pdf_bytes = io.BytesIO(r.content)
            reader = PyPDF2.PdfReader(pdf_bytes)
            text_parts = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n".join(text_parts) if text_parts else "ERROR_EXTRACT: No text extracted from PDF."
        except Exception as e:
            return f"ERROR_PDF_PARSE: {str(e)}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            if not paragraphs and soup.body:
                return soup.body.get_text(separator=" ", strip=True)[:20000]
            return "\n\n".join(paragraphs)[:20000]
        except Exception as e:
            return f"ERROR_HTML_PARSE: {str(e)}"

def summarize_text_with_gemini(text: str, max_output_chars: int = 1500) -> str:
    if not text or text.startswith("ERROR"):
        return text
    context = text[:6000]
    prompt = (
        f"Summarize the following NASA bioscience paper content in bullet points and a short summary.\n\n"
        f"Content:\n{context}\n\nOutput: 3 short bullet points of key findings, then a 2-3 sentence summary."
    )
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"ERROR_GEMINI: {str(e)}"

# --------------------------- SIDEBAR (PDF Summaries) ---------------------------
with st.sidebar:
    st.header("📂 Upload PDFs to Summarize")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded")
        for uploaded_file in uploaded_files:
            pdf_bytes = io.BytesIO(uploaded_file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
            with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
                summary = summarize_text_with_gemini(text)
            st.markdown("### 📄 Summary:")
            st.write(summary)

# --------------------------- MAIN PAGE ---------------------------

# Title
st.markdown("<h1 class='main-title'>Simplified Knowledge</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.</p>", unsafe_allow_html=True)

# Two main sections: Search (left) and Chat (right)
left_col, right_col = st.columns([2, 1])

# --------------------------- SEARCH SECTION ---------------------------
with left_col:
    st.subheader("🔎 Search Publications")
    query = st.text_input("Enter keyword to search publications (press Enter):", key="search_box", placeholder="Search NASA bioscience...")
    
    if query:
        mask = df["Title"].astype(str).str.contains(query, case=False, na=False)
        results = df[mask].reset_index(drop=True)
        st.subheader(f"Results: {len(results)} matching titles")
        if len(results) == 0:
            st.info("No matching titles. Try broader keywords.")
    else:
        results = pd.DataFrame(columns=df.columns) 

    # Display results
    for idx, row in results.iterrows():
        title = row["Title"]
        link = row["Link"]
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"**[{title}]({link})**")
        cols = st.columns([3, 1, 1])
        cols[0].write("")
        if cols[1].button("🔗 Open", key=f"open_{idx}"):
            st.markdown(f"[Open in new tab]({link})")
        if cols[2].button("Gather & Summarize", key=f"summ_{idx}"):
            with st.spinner("Gathering & extracting content..."):
                extracted = fetch_url_text(link)
            if extracted.startswith("ERROR"):
                st.error(extracted)
            else:
                st.success("Content accessed — summarizing with Gemini...")
                with st.spinner("Summarizing..."):
                    summary = summarize_text_with_gemini(extracted)
                st.markdown("**AI Summary:**")
                st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- CHAT SECTION ---------------------------
with right_col:
    st.subheader("💬 Chat with AI for quick answers!")
    q = st.text_input("Ask a question!", key="chat_box", placeholder="Type anything...")
    if q:
        try:
            resp = model.generate_content(q)  
            st.subheader("Answer:")
            st.write(resp.text)
        except Exception as e:
            st.error("AI chat failed: " + str(e))
