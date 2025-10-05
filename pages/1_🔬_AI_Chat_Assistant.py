import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---

# Configure the page
st.set_page_config(
    page_title="NASA AI Chat Assistant",
    page_icon="🔬",
    layout="centered"
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
    body {
        background-color: #0b3d91;
        color: white;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .stTextInput>div>div>input {
        color: white !important;
        background-color: #1e1e2f !important;
        border: 1px solid #444 !important;
    }
    input::placeholder {
        color: #cccccc !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS (CACHED) ---

@st.cache_data
def load_data(file_path):
    """Loads the NASA publications CSV into a pandas DataFrame."""
    try:
        # We only need Title and Link for the context
        return pd.read_csv(file_path, usecols=['Title', 'Link'])
    except FileNotFoundError:
        st.error(f"Fatal Error: The data file '{file_path}' was not found. Please make sure it's in the correct directory.")
        st.stop()
    except ValueError:
        st.error("Error: The CSV must contain 'Title' and 'Link' columns.")
        st.stop()

def find_relevant_publications(query, df, top_k=5):
    """Finds the most relevant publications from the dataframe based on a query."""
    if query:
        # A simple but effective keyword search
        mask = df["Title"].astype(str).str.contains(query, case=False, na=False)
        relevant_df = df[mask]
        return relevant_df.head(top_k)
    return pd.DataFrame() # Return empty dataframe if no query

# --- MAIN PAGE UI & LOGIC ---

st.title("🔬 AI Chat Assistant")
st.markdown("Ask me anything about the **608 NASA bioscience publications**. I'll find relevant papers and provide an informed answer.")

# Load the dataset
df = load_data("SB_publication_PMC.csv")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- GENERATE AI RESPONSE ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # 1. Find relevant context from the publications
            relevant_pubs = find_relevant_publications(prompt, df)
            
            context_str = "No specific publications found for this query."
            if not relevant_pubs.empty:
                context_str = "Based on the following relevant publications:\n"
                for _, row in relevant_pubs.iterrows():
                    context_str += f"- **Title:** {row['Title']}\n"
                
            # 2. Construct the full prompt for the Gemini model
            full_prompt = (
                "You are a specialized AI assistant with expertise in NASA's bioscience research. "
                "Your knowledge is grounded in a database of 608 publications. "
                "Answer the user's question based *only* on the context provided below. "
                "If the context is insufficient, state that you cannot find the answer in the provided publications. "
                "Be helpful, concise, and cite the titles of the papers you are referencing in your answer.\n\n"
                f"--- CONTEXT ---\n{context_str}\n\n"
                f"--- USER'S QUESTION ---\n{prompt}"
            )
            
            # 3. Generate the response
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(full_prompt)
                ai_response = response.text
            except Exception as e:
                ai_response = f"Sorry, I encountered an error. Please try again. Details: {e}"
            
            message_placeholder.markdown(ai_response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
