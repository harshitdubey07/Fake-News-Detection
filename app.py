import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Streamlit UI
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.markdown(
"""
Detect whether a news article is **Fake** or **Real** using a Machine Learning model trained on thousands of news articles.

Fill in the details below and click **Predict**.
"""
)

st.divider()

title = st.text_input("📰 News Title")

news = st.text_area(
    "📄 News Article",
    height=250
)

if st.button("Predict"):

    if title.strip() == "" or news.strip() == "":
        st.warning("Please enter both the news title and the news article.")
    else:

        content = title + " " + news

        cleaned = preprocess_text(content)

        vector = vectorizer.transform([cleaned])


        
        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        confidence = probability.max() * 100

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.progress(confidence / 100)

        st.info(f"Confidence Score: **{confidence:.2f}%**")