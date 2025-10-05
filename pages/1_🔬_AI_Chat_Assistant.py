import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---

st.set_page_config(
    page_title="NASA AI Chat Assistant",
    page_icon="🔬",
    layout="centered"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- STYLING (MATCHES MAIN PAGE) ---

st.markdown("""
    <style>
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION BAR */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* NEW THEME: White Background */
    body {
        background-color: #FFFFFF;
        color: #333333; /* Dark grey for main text */
    }
    h1 {
        color: #000000; /* Black for title */
    }
    /* Input Box */
    .stTextInput>div>div>input {
        color: #000000 !important;
        background-color: #F0F2F6 !important;
    }
    /* Link */
     a {
        color: #6A1B9A;
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION LINK ---
st.markdown("[⬅️ Back to Search](/)", unsafe_allow_html=True)


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

st.title("🔬 AI Chat Assistant")
st.markdown("Ask me anything about the **608 NASA bioscience publications**.")

df = load_data("SB_publication_PMC.csv")

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
                "If the context is insufficient, state that you cannot find the answer in the provided publications. "
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
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
