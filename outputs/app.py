import streamlit as st
import joblib
import re
import nltk
import json
import os
from nltk.corpus import stopwords
import google.generativeai as genai

# ------------------- SETUP -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Configure Gemini using environment variable or Streamlit secret
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Streamlit config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

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

# ------------------- GEMINI EXPLANATION -------------------
def get_gemini_explanation(headline, verdict):
    try:
        prompt = f"""
        A fake news detection model analyzed this headline:
        "{headline}"
        The model predicted it as: {verdict}.
        Give a short explanation in 1-2 sentences why it might be {verdict}.
        Keep it clear, factual, and easy to understand.
        """
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(Gemini explanation unavailable: {e})"

# ------------------- COMBINED FUNCTION -------------------
def predict_and_explain(headline):
    verdict, prob = predict_news(headline)
    explanation = get_gemini_explanation(headline, verdict)
    return verdict, prob, explanation

# ------------------- API MODE -------------------
query_params = st.experimental_get_query_params()
text = query_params.get("text", [None])[0]

if text:
    # act like an API endpoint
    label, prob, explanation = predict_and_explain(text)
    st.write(json.dumps({
        "headline": text,
        "prediction": label,
        "confidence": prob,
        "gemini_explanation": explanation
    }))
else:
    # ------------------- STREAMLIT UI -------------------
    st.title("📰 Fake News Detection App (Gemini Enhanced)")
    st.write("Enter a news headline to check if it’s **real or fake**. The ML model predicts, and Gemini explains why!")

    headline = st.text_area("Enter a news headline", height=100)

    if st.button("Check"):
        if headline.strip() == "":
            st.warning("Please enter a headline first.")
        else:
            label, prob, explanation = predict_and_explain(headline)
            st.markdown(f"### 🔎 Prediction: **{label}**")
            st.progress(float(prob))
            st.caption(f"Confidence: {prob:.2f}")
            st.markdown("### 🤖 Gemini Explanation:")
            st.info(explanation)

    st.markdown("---")
    st.caption("Made by Abeer Rai ✨")
