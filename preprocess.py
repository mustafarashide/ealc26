## imports
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk import word_tokenize, sent_tokenize
from nltk.stem.arlstem2 import ARLSTem2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classifcation_report
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords


def main():
    df = pd.read_csv("shared_task_train.csv")
    y = df["label"]

    df["text_without_stopwords"] = df["text"].apply(remove_arabic_stopwords)
    pd.set_option("display.max_rows", None)
    df["clean_text"] = df["text_without_stopwords"].apply(preprocess_arabic_text)
    df["features"]=df["clean_text"].apply(extract_features)
    print(df["clean_text"])
    # seeing distribution of labels (might collapse those with count <10 and feed into a different model?)
    # print(y.value_counts(normalize=False, ascending=True))

def word_to_stem(text):
    stemmed_sentence=[]
    words=wordpunct_tokenize(text)
    stemmer=ARLSTem2()
    for w in words:
        stemmed_sentence.append((stemmer.stem(w)))
    return " ".join(stemmed_sentence)

def remove_diacritics(text):
    arabic_diacritics = re.compile("r[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(arabic_diacritics, "", text)
    return text


def remove_dashes_and_qa(text):
    pattern = r"السؤال\s*-+\s*(.*?)\s*الجواب\s*-+\s*(.*)"
    return re.sub(pattern, r"\1 \2", text).strip()


def remove_arabic_stopwords(text):
    stop_words = set(stopwords.words("arabic"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)


def preprocess_arabic_text(text):
    text = remove_dashes_and_qa(text)
    text = remove_arabic_stopwords(text)
    text = remove_diacritics(text)
    text=word_to_stem(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r'[،؛؟.!"#$%&\'()*+,-/:;<=>?@[\]^_`{|}~]', "", text)
    return text


# Calculate the total length of words and divide by the number of words.
# Return 0 if no words are provided.
def average_word_length(words):
    return sum(len(word) for word in words) / len(words) if words else 0


# Calculate the ratio of unique words to total words.
# Return 0 if no words are provided.
def type_token_ratio(words):
    return len(set(words)) / len(words) if words else 0


# Split each sentence into words, calculate the average number of words per sentence.
# Return 0 if no sentences are provided.
def average_sentence_length(sentences):
    return sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0

# Count words that occur less than or equal to the frequency threshold.
def count_rare_words(words, frequency_threshold=5):
    word_freq = FreqDist(words)
    return sum(1 for word in words if word_freq[word] <= frequency_threshold)


# Return a dictionary of calculated features for the given text.
def extract_features(text):
    preprocessed_text = preprocess_arabic_text(text)
    words = word_tokenize(preprocessed_text)
    sentences = sent_tokenize(preprocessed_text)

    features = {
        'average_word_length': average_word_length(words),
        'type_token_ratio': type_token_ratio(words),
        'average_sentence_length': average_sentence_length( sent_tokenize(text)),
        'count_rare_words': count_rare_words(words)
    }

    return features
def predict(X,y):
    tfidf_vectorizer=tfidf_vectorizer(max_features=150000)
    X_tfid=tfidf_vectorizer.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_tfid,y,test_size=0.1,random_state=42,stratify=y)
    model=xgb.XGBClassifier(random_state=42,use_label_encoder=False,eval_metric="logless")
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

if __name__ == "__main__":
    main()
