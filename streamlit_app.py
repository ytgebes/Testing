import streamlit as st
import json
import time
import pandas as pd
from streamlit_extras.let_it_rain import rain
from streamlit_extras.mention import mention


LANGUAGES = {
    "Afrikaans": {"label": "🇿🇦 Afrikaans", "code": "af"},
    "العربية": {"label": "🇸🇦 العربية", "code": "ar"},
    "Հայերեն": {"label": "🇦🇲 Հայերեն", "code": "hy"},
    "Azərbaycan dili": {"label": "🇦🇿 Azərbaycan dili", "code": "az"},
    "Euskara": {"label": "🇪🇸 Euskara", "code": "eu"},
    "Беларуская": {"label": "🇧🇾 Беларуская", "code": "be"},
    "বাংলা": {"label": "🇧🇩 বাংলা", "code": "bn"},
    "Bosanski": {"label": "🇧🇦 Bosanski", "code": "bs"},
    "Български": {"label": "🇧🇬 Български", "code": "bg"},
    "Català": {"label": "🇪🇸 Català", "code": "ca"},
    "中文": {"label": "🇨🇳 中文", "code": "zh"},
    "Hrvatski": {"label": "🇭🇷 Hrvatski", "code": "hr"},
    "Čeština": {"label": "🇨🇿 Čeština", "code": "cs"},
    "Dansk": {"label": "🇩🇰 Dansk", "code": "da"},
    "Nederlands": {"label": "🇳🇱 Nederlands", "code": "nl"},
    "English": {"label": "🇺🇸 English", "code": "en"},
    "Esperanto": {"label": "🌍 Esperanto", "code": "eo"},
    "Eesti": {"label": "🇪🇪 Eesti", "code": "et"},
    "Suomi": {"label": "🇫🇮 Suomi", "code": "fi"},
    "Français": {"label": "🇫🇷 Français", "code": "fr"},
    "Galego": {"label": "🇪🇸 Galego", "code": "gl"},
    "ქართული": {"label": "🇬🇪 ქართული", "code": "ka"},
    "Deutsch": {"label": "🇩🇪 Deutsch", "code": "de"},
    "Ελληνικά": {"label": "🇬🇷 Ελληνικά", "code": "el"},
    "ગુજરાતી": {"label": "🇮🇳 ગુજરાતી", "code": "gu"},
    "Hausa": {"label": "🇳🇬 Hausa", "code": "ha"},
    "עברית": {"label": "🇮🇱 עברית", "code": "he"},
    "हिन्दी": {"label": "🇮🇳 हिन्दी", "code": "hi"},
    "Magyar": {"label": "🇭🇺 Magyar", "code": "hu"},
    "Íslenska": {"label": "🇮🇸 Íslenska", "code": "is"},
    "Bahasa Indonesia": {"label": "🇮🇩 Bahasa Indonesia", "code": "id"},
    "Gaeilge": {"label": "🇮🇪 Gaeilge", "code": "ga"},
    "Italiano": {"label": "🇮🇹 Italiano", "code": "it"},
    "日本語": {"label": "🇯🇵 日本語", "code": "ja"},
    "Basa Jawa": {"label": "🇮🇩 Basa Jawa", "code": "jv"},
    "ಕನ್ನಡ": {"label": "🇮🇳 ಕನ್ನಡ", "code": "kn"},
    "Қазақ тілі": {"label": "🇰🇿 Қазақ тілі", "code": "kk"},
    "ភាសាខ្មែរ": {"label": "🇰🇭 ភាសាខ្មែរ", "code": "km"},
    "한국어": {"label": "🇰🇷 한국어", "code": "ko"},
    "Kurdî": {"label": "🇹🇯 Kurdî", "code": "ku"},
    "Кыргызча": {"label": "🇰🇬 Кыргызча", "code": "ky"},
    "ລາວ": {"label": "🇱🇦 ລາວ", "code": "lo"},
    "Latviešu": {"label": "🇱🇻 Latviešu", "code": "lv"},
    "Lietuvių": {"label": "🇱🇹 Lietuvių", "code": "lt"},
    "Македонски": {"label": "🇲🇰 Македонски", "code": "mk"},
    "Malagasy": {"label": "🇲🇬 Malagasy", "code": "mg"},
    "Bahasa Melayu": {"label": "🇲🇾 Bahasa Melayu", "code": "ms"},
    "Malti": {"label": "🇲🇹 Malti", "code": "mt"},
    "Монгол": {"label": "🇲🇳 Монгол", "code": "mn"},
    "नेपाली": {"label": "🇳🇵 नेपाली", "code": "ne"},
    "Norsk": {"label": "🇳🇴 Norsk", "code": "no"},
    "پښتو": {"label": "🇦🇫 پښتو", "code": "ps"},
    "فارسی": {"label": "🇮🇷 فارسی", "code": "fa"},
    "Polski": {"label": "🇵🇱 Polski", "code": "pl"},
    "Português": {"label": "🇵🇹 Português", "code": "pt"},
    "ਪੰਜਾਬੀ": {"label": "🇮🇳 ਪੰਜਾਬੀ", "code": "pa"},
    "Română": {"label": "🇷🇴 Română", "code": "ro"},
    "Русский": {"label": "🇷🇺 Русский", "code": "ru"},
    "Српски": {"label": "🇷🇸 Српски", "code": "sr"},
    "Svenska": {"label": "🇸🇪 Svenska", "code": "sv"},
    "Kiswahili": {"label": "🇹🇿 Kiswahili", "code": "sw"},
    "தமிழ்": {"label": "🇮🇳 தமிழ்", "code": "ta"},
    "తెలుగు": {"label": "🇮🇳 తెలుగు", "code": "te"},
    "ไทย": {"label": "🇹🇭 ไทย", "code": "th"},
    "Türkçe": {"label": "🇹🇷 Türkçe", "code": "tr"},
    "Українська": {"label": "🇺🇦 Українська", "code": "uk"},
    "اردو": {"label": "🇵🇰 اردو", "code": "ur"},
    "O‘zbek": {"label": "🇺🇿 O‘zbek", "code": "uz"},
    "Tiếng Việt": {"label": "🇻🇳 Tiếng Việt", "code": "vi"},
    "isiXhosa": {"label": "🇿🇦 isiXhosa", "code": "xh"},
    "ייִדיש": {"label": "🇮🇱 ייִדיש", "code": "yi"},
    "Yorùbá": {"label": "🇳🇬 Yorùbá", "code": "yo"},
    "isiZulu": {"label": "🇿🇦 isiZulu", "code": "zu"},
}





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

if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}

lang_choice = st.selectbox(
    "🌐 Language",
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x]["label"],
    index=list(LANGUAGES.keys()).index(st.session_state.current_lang)
)

# For now, we will not translate UI automatically since genai is gone
translated_strings = st.session_state.translations["English"]

st.title(translated_strings["title"])
st.write(translated_strings["description"])

mention(
    label=translated_strings["mention_label"],
    icon="NASA International Space Apps Challenge",
    url="https://www.spaceappschallenge.org/"
)

uploaded_files = st.file_uploader(
    translated_strings["upload_label"], accept_multiple_files=True
)

translate_dataset = st.checkbox(translated_strings["translate_dataset_checkbox"])

if uploaded_files:
    for f in uploaded_files:
        df = pd.read_csv(f)
        st.dataframe(df)

user_input = st.text_input(translated_strings["ask_label"])
if user_input:
    st.subheader(translated_strings["response_label"])
    st.write("AI functionality removed. You typed: " + user_input)

if st.button(translated_strings["click_button"]):
    st.write(translated_strings["button_response"])

























































# The End
