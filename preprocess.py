import pandas as pd
import numpy as np
import re
import nltk
import xgboost as xgb
from nltk import FreqDist, wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.stem.arlstem2 import ARLSTem2
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def main():
    df = pd.read_csv("shared_task_train.csv")
    y = df["label"]

    print("Step 1: Preprocessing Arabic Text...")
    df["clean_text"] = df["text"].apply(preprocess_arabic_text)

    print("Step 2: Extracting Manual Features...")
    manual_features_df = pd.DataFrame(df["text"].apply(extract_features).tolist())

    scaler = StandardScaler()
    manual_features_scaled = scaler.fit_transform(manual_features_df)

    print("Step 3: Vectorizing Text...")
    vectorizer = TfidfVectorizer(max_features=50000)  # Adjusted for memory efficiency
    X_tfidf = vectorizer.fit_transform(df["clean_text"])

    X_combined = hstack([X_tfidf, manual_features_scaled])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.1, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="mlogloss")
    xgb_model.fit(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


def remove_diacritics(text):
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    return re.sub(arabic_diacritics, "", text)


def remove_arabic_stopwords(text):
    stop_words = set(stopwords.words("arabic"))
    word_tokens = word_tokenize(text)
    return " ".join([w for w in word_tokens if w not in stop_words])


def preprocess_arabic_text(text):
    pattern = r"السؤال\s*-+\s*(.*?)\s*الجواب\s*-+\s*(.*)"
    text = re.sub(pattern, r"\1 \2", text).strip()

    text = remove_arabic_stopwords(text)
    text = remove_diacritics(text)

    stemmer = ARLSTem2()
    words = wordpunct_tokenize(text)
    text = " ".join([stemmer.stem(w) for w in words])

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_features(text):
    words_raw = word_tokenize(text)
    sentences = sent_tokenize(text)

    avg_word_len = sum(len(w) for w in words_raw) / len(words_raw) if words_raw else 0
    ttr = len(set(words_raw)) / len(words_raw) if words_raw else 0
    avg_sent_len = (
        sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    )

    return {
        "avg_word_length": avg_word_len,
        "type_token_ratio": ttr,
        "avg_sentence_length": avg_sent_len,
        "word_count": len(words_raw),
    }


if __name__ == "__main__":
    main()
