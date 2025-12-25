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
    # seeing distribution of labels (might collapse those with count <10 and feed into a different model?)
    # print(y.value_counts(normalize=False, ascending=True))


def remove_diacritics(text):
    arabic_diacritics = re.compile("r[\u0617-\u061A\u064B-\u0625]")
    text=re.sub(arabic_diacritics,'',text)

def remove_dashes_and_qa(text):
    pattern=
    

def remove_arabic_stopwords(text):
    stop_words = set(stopwords.words("arabic"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)


if __name__ == "__main__":
    main()
