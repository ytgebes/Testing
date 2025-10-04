import streamlit as st
import pandas as pd
from streamlit_extras.let_it_rain import rain
from streamlit_extras.mention import mention

# Language dictionary
LANGUAGES = {
    "Afrikaans": {"label": "🇿🇦 Afrikaans (Afrikaans)", "code": "af"},
    "العربية": {"label": "🇸🇦 العربية (Arabic)", "code": "ar"},
    "English": {"label": "🇺🇸 English (English)", "code": "en"},
    "Türkçe": {"label": "🇹🇷 Türkçe (Turkish)", "code": "tr"},
    # ... add all other languages as needed
}

# UI strings
UI_STRINGS_EN = {
    "title": "Simplified Knowledge",
    "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "upload_label": "Upload CSV data",
    "ask_label": "Ask anything:",
    "response_label": "Response:",
    "click_button": "Click here, nothing happens",
    "translate_dataset_checkbox": "Translate dataset column names (may take time)",
    "mention_label": "Official NASA Website",
    "button_response": "Hooray"
}

# Initialize session state
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}

# Ensure default index exists
lang_keys = list(LANGUAGES.keys())
default_index = lang_keys.index(st.session_state.current_lang) if st.session_state.current_lang in lang_keys else 0

# Language selectbox
lang_choice = st.selectbox(
    "🌐 Language",
    options=lang_keys,
    format_func=lambda x: LANGUAGES[x]["label"],
    index=default_index
)
st.session_state.current_lang = lang_choice

# Use English strings for now
translated_strings = st.session_state.translations["English"]

# UI components
st.title(translated_strings["title"])
st.write(translated_strings["description"])

mention(
    label=translated_strings["mention_label"],
    url="https://www.spaceappschallenge.org/"
)

uploaded_files = st.file_uploader(
    translated_strings["upload_label"], accept_multiple_files=True
)

translate_dataset = st.checkbox(translated_strings["translate_dataset_checkbox"])

if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")

user_input = st.text_input(translated_strings["ask_label"])
if user_input:
    st.subheader(translated_strings["response_label"])
    st.write("AI functionality removed. You typed: " + user_input)

if st.button(translated_strings["click_button"]):
    st.write(translated_strings["button_response"])
