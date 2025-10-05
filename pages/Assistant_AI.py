import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Assistant AI", page_icon="💬", layout="wide")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (This is the hamburger menu/sidebar) */
    [data-testid="stSidebar"] { display: none; }
    /* This also hides the auto-generated navigation menu */
    [data-testid="stPageLink"] { display: none; } 
    
    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }
    
    /* Remove custom nav button styling as we now use st.navigation */
    .nav-container { display: none; } 
    
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
    try:
        # Assuming the CSV is in the root directory relative to where the app is run
        return pd.read_csv("SB_publication_PMC.csv", usecols=['Title', 'Link']) 
    except (FileNotFoundError, ValueError):
        st.error("Error: Could not load the publication data file (SB_publication_PMC.csv).")
        st.stop()

def find_relevant_publications(query, df, top_k=5):
    if query:
        mask = df["Title"].astype(str).str.lower().str.contains(query.lower(), na=False)
        return df[mask].head(top_k)
    return pd.DataFrame()

# --- MAIN PAGE UI & LOGIC ---

# Load data once
df = load_data("SB_publication_PMC.csv")

# NOTE: Removed the st.markdown for the "⬅️ Back to Search" button as st.navigation handles it.

st.title("Assistant AI")
st.markdown("<p style='text-align: center;'>Ask me anything about the **608 NASA bioscience publications**.</p>", unsafe_allow_html=True)

# Centralize the chat interface
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
                
                context_str = "No specific publications found matching the search terms in the question."
                if not relevant_pubs.empty:
                    context_str = "Based on the following relevant publications:\n"
                    for _, row in relevant_pubs.iterrows():
                        context_str += f"- **Title:** {row['Title']}\n  - *Link:* {row['Link']}\n"
                
                full_prompt = (
                    "You are a specialized AI assistant for NASA's bioscience research. "
                    "Answer the user's question based *only* on the context provided below. "
                    "If the context is insufficient, state clearly that you cannot find the answer based on the publications. "
                    "For a helpful answer, explicitly cite the **titles** of the papers you reference in your response.\n\n"
                    f"--- CONTEXT (Relevant Publication Titles) ---\n{context_str}\n\n"
                    f"--- USER'S QUESTION ---\n{prompt}"
                )
                
                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                except Exception as e:
                    ai_response = f"Sorry, an error occurred with the AI service: {e}"
                
                placeholder.markdown(ai_response)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
