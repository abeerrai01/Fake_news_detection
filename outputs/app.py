import streamlit as st
import joblib
import re
import nltk
import json
import os
import base64
import requests
from nltk.corpus import stopwords
import google.generativeai as genai

# ------------------- SETUP -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(
    page_title="Fake News & Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)

# ------------------- BACKGROUND IMAGE -------------------
def set_background(image_file):
    abs_path = os.path.join(os.path.dirname(__file__), image_file)
    if not os.path.exists(abs_path):
        st.warning(f"⚠️ Background image not found: {abs_path}")
        return

    with open(abs_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded_image}") no-repeat center center fixed;
                background-size: cover;
                color: white;
                font-family: 'Poppins', sans-serif;
            }}

            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0, 0, 0, 0.45);
                z-index: 0;
            }}

            .stApp * {{
                position: relative;
                z-index: 1;
            }}

            .block-container {{
                background-color: transparent !important;
            }}

            div.stButton > button {{
                background-color: #ffcc00;
                color: #000;
                border-radius: 12px;
                padding: 0.7em 1.5em;
                font-weight: 600;
                border: none;
                transition: 0.3s;
            }}
            div.stButton > button:hover {{
                background-color: #ffaa00;
                color: white;
                transform: scale(1.05);
            }}

            textarea {{
                background-color: rgba(0,0,0,0.5) !important;
                color: white !important;
                border-radius: 12px !important;
                border: 1px solid #ffcc00 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# ------------------- LOAD FAKE NEWS MODEL -------------------
@st.cache_resource
def load_resources():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/fake_news_model.pkl")
    return tfidf, model

tfidf, model = load_resources()

# ------------------- TEXT CLEANING -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text


# ------------------- PREDICT FAKE NEWS -------------------
def predict_news(headline):
    clean = clean_text(headline)
    features = tfidf.transform([clean])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    label = "✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS"
    return label, float(prob)


# ------------------- GEMINI EXPLANATION -------------------
def stream_gemini_explanation(headline, verdict):
    try:
        prompt = f"""
        A fake news detection model analyzed this headline:
        "{headline}"
        The model predicted it as: {verdict}.
        Explain in 1–2 short sentences why it could be {verdict}.
        """

        model_g = genai.GenerativeModel("gemini-2.5-flash")
        response = model_g.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"(Gemini explanation unavailable: {e})"


# ------------------- PHISHING URL CHECKER (NEW API FORMAT) -------------------
def check_phishing_url(url):
    try:
        payload = {"url": url}
        res = requests.post(
            "https://theabeerrai-url-phishing.hf.space/predict",
            json=payload,
            timeout=10
        )

        if res.status_code != 200:
            return "⚠️ Error reaching API", 0.0, None

        data = res.json()

        pred = data.get("prediction", 0)          # 1 = phishing, 0 = safe
        prob = float(data.get("probability", 0))  # confidence
        details = data.get("details", {})         # feature breakdown

        if pred == 1:
            label = "❌ PHISHING URL"
        else:
            label = "✅ SAFE URL"

        return label, prob, details

    except Exception as e:
        return f"⚠️ API Error: {e}", 0.0, None


# ------------------- UI -------------------
st.markdown("<h1>🛡️ Fake News & URL Phishing Detector</h1>", unsafe_allow_html=True)
st.write("Analyze headlines *and* URLs with machine learning + Gemini explanations.")

tabs = st.tabs(["📰 Fake News Checker", "🔐 URL Phishing Checker"])


# ------------------- TAB 1: FAKE NEWS -------------------
with tabs[0]:
    headline = st.text_area("Enter a news headline", height=100)

    if st.button("Check Headline"):
        if headline.strip() == "":
            st.warning("Please enter a headline.")
        else:
            label, prob = predict_news(headline)

            # Result box
            if "REAL" in label:
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            st.progress(prob)
            st.caption(f"Confidence: {prob:.2f}")

            # Gemini explanation
            st.markdown("### 💬 Gemini Explanation")
            st.markdown(
                "<div style='background-color:rgba(0,0,0,0.6); padding:10px; "
                "border-left:4px solid #ffcc00; border-radius:8px;'>",
                unsafe_allow_html=True,
            )
            st.write_stream(stream_gemini_explanation(headline, label))
            st.markdown("</div>", unsafe_allow_html=True)


# ------------------- TAB 2: PHISHING URL CHECKER -------------------
with tabs[1]:
    url = st.text_input("Enter a URL")

    if st.button("Check URL"):
        if url.strip() == "":
            st.warning("Please enter a valid URL.")
        else:
            label, prob, details = check_phishing_url(url)

            if "SAFE" in label:
                st.success(f"URL Status: {label}")
            else:
                st.error(f"URL Status: {label}")

            st.progress(prob)
            st.caption(f"Confidence: {prob:.2f}")

            # Feature breakdown
            if details:
                st.markdown("### 🔍 URL Feature Breakdown")
                st.json(details)


# ------------------- FOOTER -------------------
st.markdown(
    "<div style='text-align:center; color:gray;'>Made with ❤️ by <b>Abeer Rai</b> | Powered by Gemini ✨</div>",
    unsafe_allow_html=True,
)
