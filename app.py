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

st.title("📰 Fake News Detection")

st.write(
    "Enter a news article below and the trained Machine Learning model will predict whether it is **Fake** or **Real**."
)

news = st.text_area(
    "Enter News Text",
    height=250
)

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        cleaned = preprocess_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")