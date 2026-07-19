import pandas as pd
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download("stopwords")

# Initialize
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()

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


# -----------------------------
# Load Dataset
# -----------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Loaded Successfully!")


# -----------------------------
# Combine Title + Text
# -----------------------------
df["content"] = df["title"] + " " + df["text"]

print("Content Created!")


# -----------------------------
# Clean Text
# -----------------------------
df["clean_text"] = df["content"].apply(preprocess_text)

print("Text Preprocessing Completed!")


# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["clean_text"])

y = df["label"]

print("TF-IDF Completed!")


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(rf, "models/fake_news_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")