import numpy as np
from sklearn.datasets import fetch_20newsgroups



train_data = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers','quotes','footers'))

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target

print(len(X_train),len(X_test))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # usuń cyfry i znaki specjalne
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_train_clean = [clean_text(doc) for doc in X_train]
X_test_clean = [clean_text(doc) for doc in X_test]


import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re
from nltk.tokenize import word_tokenize
import nltk


nltk.download("punkt")


train_data = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers','quotes','footers'))

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target

print(len(X_train), len(X_test))


def clean_and_tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # usuń cyfry i znaki specjalne
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)  # tokenizacja
    return tokens


X_train_tokens = [clean_and_tokenize(doc) for doc in X_train]
X_test_tokens = [clean_and_tokenize(doc) for doc in X_test]


print("Przykład tokenów z 1 dokumentu:", X_train_tokens[0][:30])