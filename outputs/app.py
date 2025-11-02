import streamlit as st
import joblib
import re
import nltk
import json
import os
import base64
from nltk.corpus import stopwords
import google.generativeai as genai

# ------------------- SETUP -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Configure Gemini API key (from secrets or env)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ------------------- BACKGROUND IMAGE SETUP -------------------
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
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                background-repeat: no-repeat;
                color: white;
                font-family: 'Poppins', sans-serif;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.6);
                z-index: 0;
            }}
            .stApp > * {{
                position: relative;
                z-index: 1;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👇 Use your local image here (ensure it's uploaded to GitHub before deploying)
set_background("background.png")

# ------------------- CUSTOM STYLING -------------------
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #ffcc00 !important;
            font-weight: 700;
        }
        div.stButton > button {
            background-color: #ffcc00;
            color: #000;
            border-radius: 12px;
            padding: 0.7em 1.5em;
            font-weight: 600;
            transition: 0.3s;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #ffaa00;
            color: white;
            transform: scale(1.05);
        }
        textarea {
            background-color: rgba(0, 0, 0, 0.5) !important;
            color: #fff !important;
            border-radius: 12px !important;
            border: 1px solid #ffcc00 !important;
        }
        .stProgress > div > div > div > div {
            background-color: #ffcc00;
        }
        .stMarkdown h3 {
            color: #ffcc00;
        }
        .stMarkdown p, .stCaption {
            color: #dddddd !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
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

# ------------------- PREDICTION -------------------
def predict_news(headline):
    clean = clean_text(headline)
    features = tfidf.transform([clean])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    label = "✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS"
    return label, float(prob)

# ------------------- STREAM GEMINI EXPLANATION -------------------
def stream_gemini_explanation(headline, verdict):
    try:
        prompt = f"""
        A fake news detection model analyzed this headline:
        "{headline}"
        The model predicted it as: {verdict}.
        Give a short explanation (1–2 sentences) why it might be {verdict}.
        Be factual, clear, and concise.
        """

        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = model_gemini.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"(Gemini explanation unavailable: {e})"

# ------------------- COMBINED FUNCTION -------------------
def predict_and_explain(headline):
    verdict, prob = predict_news(headline)
    return verdict, prob

# ------------------- API MODE -------------------
query_params = st.experimental_get_query_params()
text = query_params.get("text", [None])[0]

if text:
    label, prob = predict_news(text)
    explanation = "".join(stream_gemini_explanation(text, label))
    st.write(json.dumps({
        "headline": text,
        "prediction": label,
        "confidence": prob,
        "gemini_explanation": explanation
    }))

else:
    st.markdown("<h1>📰 Fake News Detection App (Gemini Enhanced)</h1>", unsafe_allow_html=True)
    st.write("Enter a headline to check if it’s **real or fake**. The ML model predicts instantly — then Gemini explains why! 🤖")

    headline = st.text_area("Enter a news headline", height=100)

    if st.button("Check"):
        if headline.strip() == "":
            st.warning("Please enter a headline first.")
        else:
            label, prob = predict_and_explain(headline)
            if "REAL" in label:
                st.success(f"✅ **Prediction:** {label}")
            else:
                st.error(f"❌ **Prediction:** {label}")

            st.progress(float(prob))
            st.caption(f"Confidence: {prob:.2f}")

            st.markdown("### 💬 Gemini Explanation:")
            with st.container():
                st.markdown(
                    "<div style='background-color:rgba(0,0,0,0.6); padding:10px; border-left:4px solid #ffcc00; border-radius:8px;'>",
                    unsafe_allow_html=True,
                )
                st.write_stream(stream_gemini_explanation(headline, label))
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray;'>Made with ❤️ by <b>Abeer Rai</b> | Powered by Gemini ✨</div>",
        unsafe_allow_html=True,
    )
