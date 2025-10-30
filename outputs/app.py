import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your saved model and vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/fake_news_model.pkl")

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# Function to predict
def predict_news(headline):
    clean = clean_text(headline)
    features = tfidf.transform([clean])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    label = "✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS"
    return label, prob

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline below to check if it’s **real** or **fake** using an AI model trained on Indian fake news datasets.")

headline = st.text_area("Enter a news headline", height=100)

if st.button("Check"):
    if headline.strip() == "":
        st.warning("Please enter a headline first.")
    else:
        label, prob = predict_news(headline)
        st.markdown(f"### 🔎 Prediction: **{label}**")
        st.progress(float(prob))
        st.caption(f"Confidence: {prob:.2f}")

st.markdown("---")
st.caption("Made by Abeer Rai ✨")
