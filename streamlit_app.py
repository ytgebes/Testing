import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Assistant AI", page_icon="💬", layout="wide")

try:
    # Ensure you have the GEMINI_API_KEY in your Streamlit secrets
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI. Make sure 'GEMINI_API_KEY' is set in st.secrets: {e}")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION */
    [data-testid="stSidebar"] { display: none; }

    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }
    
    /* Nav button container aligned to the left */
    .nav-container {
        display: flex;
        justify-content: flex-start; /* Aligns button to the left */
        padding-top: 1.5rem; /* PADDING TO MATCH THE MAIN PAGE VISUALLY */
        padding-bottom: 1rem;
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

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file_path):
    """Loads the publication data for the RAG search."""
    try:
        return pd.read_csv(file_path, usecols=['Title', 'Link'])
    except (FileNotFoundError, ValueError):
        st.error("Error: Could not load the publication data file.")
        st.stop()

def find_relevant_publications(query, df, top_k=5):
    """Finds publications whose titles contain the query string."""
    if query:
        # Use .str.lower() for more robust case-insensitive matching
        mask = df["Title"].astype(str).str.lower().str.contains(query.lower(), na=False)
        return df[mask].head(top_k)
    return pd.DataFrame()

# --- MAIN PAGE UI & LOGIC ---

# Load data once
df = load_data("SB_publication_PMC.csv")

# --- NAVIGATION BUTTON ---
# Assumes the main file is named Home.py or is the default app file
st.markdown(
    '<div class="nav-container"><div class="nav-button"><a href="/" target="_self">⬅️ Back to Search</a></div></div>',
    unsafe_allow_html=True
)

st.title("Assistant AI")
st.markdown("<p style='text-align: center;'>Ask me anything about the **608 NASA bioscience publications**.</p>", unsafe_allow_html=True)

# Centralize the chat interface
_, col2, _ = st.columns([1, 2, 1])
with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Searching publications and formulating answer..."):
                
                # 1. Retrieval Step (RAG)
                relevant_pubs = find_relevant_publications(prompt, df)
                
                context_str = "No specific publications found matching the search terms in the question."
                if not relevant_pubs.empty:
                    context_str = "Based on the following relevant publications:\n"
                    # Include the titles and links for context and citation
                    for _, row in relevant_pubs.iterrows():
                        context_str += f"- **Title:** {row['Title']}\n  - *Link:* {row['Link']}\n"
                
                # 2. Generation Step
                full_prompt = (
                    "You are a specialized AI assistant for NASA's bioscience research. "
                    "Answer the user's question based *only* on the context provided below. "
                    "If the context is insufficient, state clearly that you cannot find the answer based on the publications. "
                    "For a helpful answer, explicitly cite the **titles** of the papers you reference in your response.\n\n"
                    f"--- CONTEXT (Relevant Publication Titles) ---\n{context_str}\n\n"
                    f"--- USER'S QUESTION ---\n{prompt}"
                )
                
                try:
                    client = genai.Client()
                    response = client.models.generate_content(
                        model=MODEL_NAME, 
                        contents=full_prompt
                    )
                    ai_response = response.text
                except Exception as e:
                    ai_response = f"Sorry, an error occurred with the AI service: {e}"
                
                placeholder.markdown(ai_response)
        
        # Add assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
