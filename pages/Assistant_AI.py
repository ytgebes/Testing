import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Assistant AI", page_icon="💬", layout="wide")

try:
    # UPDATED: Model name changed to gemini-1.5-pro
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-1.5-pro"
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
    }
    .nav-button a {
        background-color: #6c757d; color: white; padding: 10px 20px;
        border-radius: 8px; text-decoration: none; font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .nav-button a:hover { background-color: #5a6268; }

    /* Main Theme */
    body { background-color: #FFFFFF; color: #333333; }
    h1 { color: #000000; text-align: center; }
    .stTextInput>div>div>input { color: #000000 !important; background-color: #F0F2F6 !important; }
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION BUTTON (MOVED TO TOP-LEFT) ---
st.markdown(
    '<div class="nav-container"><div class="nav-button"><a href="/" target="_self">⬅️ Back to Search</a></div></div>',
    unsafe_allow_html=True
)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, usecols=['Title', 'Link'])
    except (FileNotFoundError, ValueError):
        st.error("Error: Could not load the publication data file.")
        st.stop()

def find_relevant_publications(query, df, top_k=5):
    if query:
        mask = df["Title"].astype(str).str.contains(query, case=False, na=False)
        return df[mask].head(top_k)
    return pd.DataFrame()

# --- MAIN PAGE UI & LOGIC ---
st.title("Assistant AI")
st.markdown("<p style='text-align: center;'>Ask me anything about the <strong>608 NASA bioscience publications</strong>.</p>", unsafe_allow_html=True)

_, col2, _ = st.columns([1, 2, 1])
with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Searching publications and formulating answer..."):
                relevant_pubs = find_relevant_publications(prompt, df)
                
                context_str = "No specific publications found for this query."
                if not relevant_pubs.empty:
                    context_str = "Based on the following relevant publications:\n"
                    for _, row in relevant_pubs.iterrows():
                        context_str += f"- **Title:** {row['Title']}\n"
                
                full_prompt = (
                    "You are a specialized AI assistant for NASA's bioscience research. "
                    "Answer the user's question based *only* on the context provided below. "
                    "If the context is insufficient, state that you cannot find the answer. "
                    "Cite the titles of papers you reference.\n\n"
                    f"--- CONTEXT ---\n{context_str}\n\n"
                    f"--- USER'S QUESTION ---\n{prompt}"
                )
                
                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                except Exception as e:
                    ai_response = f"Sorry, an error occurred: {e}"
                
                placeholder.markdown(ai_response)
        
        st.session_state.messages.append({"role": "assistant", content: ai_response})
