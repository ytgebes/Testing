import streamlit as st
import json
import time
import pandas as pd
from streamlit_extras.let_it_rain import rain
from streamlit_extras.mention import mention


LANGUAGES = {
    "Afrikaans": {"label": "🇿🇦 Afrikaans (Afrikaans)", "code": "af"},
    "العربية": {"label": "🇸🇦 العربية (Arabic)", "code": "ar"},
    "Հայերեն": {"label": "🇦🇲 Հայերեն (Armenian)", "code": "hy"},
    "Azərbaycan dili": {"label": "🇦🇿 Azərbaycan dili (Azerbaijani)", "code": "az"},
    "Euskara": {"label": "🇪🇸 Euskara (Basque)", "code": "eu"},
    "Беларуская": {"label": "🇧🇾 Беларуская (Belarusian)", "code": "be"},
    "বাংলা": {"label": "🇧🇩 বাংলা (Bengali)", "code": "bn"},
    "Bosanski": {"label": "🇧🇦 Bosanski (Bosnian)", "code": "bs"},
    "Български": {"label": "🇧🇬 Български (Bulgarian)", "code": "bg"},
    "Català": {"label": "🇪🇸 Català (Catalan)", "code": "ca"},
    "中文": {"label": "🇨🇳 中文 (Chinese)", "code": "zh"},
    "Hrvatski": {"label": "🇭🇷 Hrvatski (Croatian)", "code": "hr"},
    "Čeština": {"label": "🇨🇿 Čeština (Czech)", "code": "cs"},
    "Dansk": {"label": "🇩🇰 Dansk (Danish)", "code": "da"},
    "Nederlands": {"label": "🇳🇱 Nederlands (Dutch)", "code": "nl"},
    "English": {"label": "🇺🇸 English (English)", "code": "en"},
    "Eesti": {"label": "🇪🇪 Eesti (Estonian)", "code": "et"},
    "Suomi": {"label": "🇫🇮 Suomi (Finnish)", "code": "fi"},
    "Français": {"label": "🇫🇷 Français (French)", "code": "fr"},
    "Galego": {"label": "🇪🇸 Galego (Galician)", "code": "gl"},
    "ქართული": {"label": "🇬🇪 ქართული (Georgian)", "code": "ka"},
    "Deutsch": {"label": "🇩🇪 Deutsch (German)", "code": "de"},
    "Ελληνικά": {"label": "🇬🇷 Ελληνικά (Greek)", "code": "el"},
    "ગુજરાતી": {"label": "🇮🇳 ગુજરાતી (Gujarati)", "code": "gu"},
    "Hausa": {"label": "🇳🇬 Hausa (Hausa)", "code": "ha"},
    "עברית": {"label": "🇮🇱 עברית (Hebrew)", "code": "he"},
    "हिन्दी": {"label": "🇮🇳 हिन्दी (Hindi)", "code": "hi"},
    "Magyar": {"label": "🇭🇺 Magyar (Hungarian)", "code": "hu"},
    "Íslenska": {"label": "🇮🇸 Íslenska (Icelandic)", "code": "is"},
    "Bahasa Indonesia": {"label": "🇮🇩 Bahasa Indonesia (Indonesian)", "code": "id"},
    "Gaeilge": {"label": "🇮🇪 Gaeilge (Irish)", "code": "ga"},
    "Italiano": {"label": "🇮🇹 Italiano (Italian)", "code": "it"},
    "日本語": {"label": "🇯🇵 日本語 (Japanese)", "code": "ja"},
    "Basa Jawa": {"label": "🇮🇩 Basa Jawa (Javanese)", "code": "jv"},
    "ಕನ್ನಡ": {"label": "🇮🇳 ಕನ್ನಡ (Kannada)", "code": "kn"},
    "Қазақ тілі": {"label": "🇰🇿 Қазақ тілі (Kazakh)", "code": "kk"},
    "ភាសាខ្មែរ": {"label": "🇰🇭 ភាសាខ្មែរ (Khmer)", "code": "km"},
    "한국어": {"label": "🇰🇷 한국어 (Korean)", "code": "ko"},
    "Kurdî": {"label": "🇹🇯 Kurdî (Kurdish)", "code": "ku"},
    "Кыргызча": {"label": "🇰🇬 Кыргызча (Kyrgyz)", "code": "ky"},
    "ລາວ": {"label": "🇱🇦 ລາວ (Lao)", "code": "lo"},
    "Latviešu": {"label": "🇱🇻 Latviešu (Latvian)", "code": "lv"},
    "Lietuvių": {"label": "🇱🇹 Lietuvių (Lithuanian)", "code": "lt"},
    "Македонски": {"label": "🇲🇰 Македонски (Macedonian)", "code": "mk"},
    "Malagasy": {"label": "🇲🇬 Malagasy (Malagasy)", "code": "mg"},
    "Bahasa Melayu": {"label": "🇲🇾 Bahasa Melayu (Malay)", "code": "ms"},
    "Malti": {"label": "🇲🇹 Malti (Maltese)", "code": "mt"},
    "Монгол": {"label": "🇲🇳 Монгол (Mongolian)", "code": "mn"},
    "नेपाली": {"label": "🇳🇵 नेपाली (Nepali)", "code": "ne"},
    "Norsk": {"label": "🇳🇴 Norsk (Norwegian)", "code": "no"},
    "پښتو": {"label": "🇦🇫 پښتو (Pashto)", "code": "ps"},
    "فارسی": {"label": "🇮🇷 فارسی (Persian)", "code": "fa"},
    "Polski": {"label": "🇵🇱 Polski (Polish)", "code": "pl"},
    "Português": {"label": "🇵🇹 Português (Portuguese)", "code": "pt"},
    "ਪੰਜਾਬੀ": {"label": "🇮🇳 ਪੰਜਾਬੀ (Punjabi)", "code": "pa"},
    "Română": {"label": "🇷🇴 Română (Romanian)", "code": "ro"},
    "Русский": {"label": "🇷🇺 Русский (Russian)", "code": "ru"},
    "Српски": {"label": "🇷🇸 Српски (Serbian)", "code": "sr"},
    "Svenska": {"label": "🇸🇪 Svenska (Swedish)", "code": "sv"},
    "Kiswahili": {"label": "🇹🇿 Kiswahili (Swahili)", "code": "sw"},
    "தமிழ்": {"label": "🇮🇳 தமிழ் (Tamil)", "code": "ta"},
    "తెలుగు": {"label": "🇮🇳 తెలుగు (Telugu)", "code": "te"},
    "ไทย": {"label": "🇹🇭 ไทย (Thai)", "code": "th"},
    "Türkçe": {"label": "🇹🇷 Türkçe (Turkish)", "code": "tr"},
    "Українська": {"label": "🇺🇦 Українська (Ukrainian)", "code": "uk"},
    "اردو": {"label": "🇵🇰 اردو (Urdu)", "code": "ur"},
    "O‘zbek": {"label": "🇺🇿 O‘zbek (Uzbek)", "code": "uz"},
    "Tiếng Việt": {"label": "🇻🇳 Tiếng Việt (Vietnamese)", "code": "vi"},
    "isiXhosa": {"label": "🇿🇦 isiXhosa (Xhosa)", "code": "xh"},
    "ייִדיש": {"label": "🇮🇱 ייִדיש (Yiddish)", "code": "yi"},
    "Yorùbá": {"label": "🇳🇬 Yorùbá (Yoruba)", "code": "yo"},
    "isiZulu": {"label": "🇿🇦 isiZulu (Zulu)", "code": "zu"},
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
