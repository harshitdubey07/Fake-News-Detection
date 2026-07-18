# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Fake News Detection System that classifies news articles as **Fake** or **Real**.

The model is trained on a real-world dataset containing fake and genuine news articles. Various Machine Learning algorithms were implemented and compared using multiple evaluation metrics.

A Streamlit web application has been developed to allow users to enter news articles and receive instant predictions.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Text Preprocessing
- TF-IDF Feature Extraction
- Multiple Machine Learning Models
- Model Performance Comparison
- Interactive Streamlit Web App
- Fake/Real News Prediction

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Linear Support Vector Machine (SVM)

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── news.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone <repository-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📷 Screenshots

(Add screenshots of your Streamlit application here after deployment.)

---

## 📌 Future Improvements

- Deep Learning models (LSTM/BERT)
- Explainable AI using LIME or SHAP
- Confidence score for predictions
- News URL verification

---

## 👨‍💻 Author

**Harshit Dubey**

Machine Learning Enthusiast