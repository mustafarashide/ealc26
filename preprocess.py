## imports
from nltk import FreqDist
from nltk import word_tokenize, sent_tokenize
import pandas as pd
import nltk
import re

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords


def main():
    df = pd.read_csv("shared_task_train.csv")
    y = df["label"]

    df["text_without_stopwords"] = df["text"].apply(remove_arabic_stopwords)
    pd.set_option("display.max_rows", None)
    print(df["text_without_stopwords"])
    df["clean_text"] = df["text_without_stopwords"].apply(preprocess_arabic_text)
    print(df["clean_text"])
    # seeing distribution of labels (might collapse those with count <10 and feed into a different model?)
    # print(y.value_counts(normalize=False, ascending=True))


def remove_diacritics(text):
    arabic_diacritics = re.compile("r[\u0617-\u061a\u064b-\u0625]")
    text = re.sub(arabic_diacritics, "", text)


def remove_dashes_and_qa(text):
    pattern = r"السؤال\s*-+\s*(.*?)\s*الجواب\s*-+\s*(.*)"
    return re.sub(pattern, r"\1 \2", text).strip()


def remove_arabic_stopwords(text):
    stop_words = set(stopwords.words("arabic"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)


def preprocess_arabic_text(text):
    text = remove_diacritics(text)
    text = remove_dashes_and_qa(text)
    text = remove_arabic_stopwords(text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\u600-\u06FF\s]", "", text)
    text = re.sub(r'[،؛؟.!"#$%&\'()*+,-/:;<=>?@[\]^_`{|}~]', "", text)
    return text


if __name__ == "__main__":
    main()
