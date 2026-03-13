"""
NLP Project 1 - Hate Speech Classification Pipeline
Experimental pipeline for comparing preprocessing, feature extraction, and models.

Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Author: Sarruja Sabesan and Natalie Jakab
Date: 08.03.2026
Modul: Natural Language Processing (NLP)
"""

import re
import csv
import sys
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

random.seed(42)
np.random.seed(42)

STOPWORDS = stopwords.words('english')
STEMMER = SnowballStemmer("english")
LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text, remove_stopwords=True, remove_numbers=True, do_stem=True, do_lemmatize=False):
    """
    Tokenize and optionally apply stopword removal, stemming, or lemmatization.
    """
    tokens = re.findall(r'\w+', text.lower())
    if remove_numbers:
        tokens = [w for w in tokens if not re.sub(r'[\'\.,]', '', w).isdigit()]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS]
    if do_stem:
        tokens = [STEMMER.stem(w) for w in tokens]
    elif do_lemmatize:
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def load_and_preprocess(csv_path='data/train.csv', preprocess_fn=None, cache_path=None, reprocess=False):
    """
    Load train.csv and apply preprocessing.
    Uses pickle cache to avoid reprocessing every time.

    Args:
        csv_path:       path to train.csv
        preprocess_fn:  function that takes a string and returns a string
        cache_path:     path to save/load pickle cache (optional)
        reprocess:      if True, ignore cache and reprocess

    Returns:
        X (list of str), Y (list of int)
    """
    if cache_path and not reprocess:
        try:
            X = pickle.load(open(cache_path + '_X.pkl', 'rb'))
            Y = pickle.load(open(cache_path + '_Y.pkl', 'rb'))
            print(f"Loaded from cache: {cache_path}")
            return X, Y
        except FileNotFoundError:
            pass

    X, Y = [], []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            sys.stderr.write(f'\rProcessing row {i}')
            sys.stderr.flush()
            text = row[1]
            if preprocess_fn:
                text = preprocess_fn(text)
            else:
                text = preprocess(text)
            label = 1 if '1' in row[2:] else 0
            X.append(text)
            Y.append(label)
    sys.stderr.write('\n')

    original_len = len(X)
    seen = set()
    X_dedup, Y_dedup = [], []
    for text, label in zip(X, Y):
        if text not in seen:
            seen.add(text)
            X_dedup.append(text)
            Y_dedup.append(label)
    X, Y = X_dedup, Y_dedup

    if cache_path:
        pickle.dump(X, open(cache_path + '_X.pkl', 'wb'))
        pickle.dump(Y, open(cache_path + '_Y.pkl', 'wb'))
        print(f"Saved to cache: {cache_path}")

    print(f"Duplicates removed: {original_len - len(X)} | Remaining: {len(X)} | Distribution: {Counter(Y)}")
    return X, Y


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────

def get_vectorizer(method='tfidf', max_features=1000):
    """
    Returns a sklearn vectorizer.

    Args:
        method:       'tfidf' or 'bow'
        max_features: vocabulary size

    Returns:
        vectorizer instance (not yet fitted)
    """
    if method == 'tfidf':
        return TfidfVectorizer(max_features=max_features)
    elif method == 'bow':
        return CountVectorizer(max_features=max_features)
    else:
        raise ValueError(f"Unknown vectorizer method: {method}. Use 'tfidf' or 'bow'.")


# ─────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────

def get_model(name='svm', C=1.0):
    """
    Returns a sklearn classifier.

    Args:
        name: 'svm', 'logreg', or 'naivebayes'
        C:    regularization parameter for SVM and LogReg (default: 1.0)

    Returns:
        classifier instance
    """
    if name == 'svm':
        return SVC(C=C, class_weight='balanced', random_state=42)
    elif name == 'logreg':
        return LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    elif name == 'naivebayes':
        return MultinomialNB()
    else:
        raise ValueError(f"Unknown model: {name}. Use 'svm', 'logreg', or 'naivebayes'.")


# ─────────────────────────────────────────────
# 4. RUN ONE EXPERIMENT
# ─────────────────────────────────────────────

def run_experiment(X, Y, vectorizer, model, downsample=False, test_size=0.2, verbose=True):
    """
    Vectorize, split, train, and evaluate one experiment.

    Args:
        X:           list of preprocessed texts
        Y:           list of labels (0/1)
        vectorizer:  fitted or unfitted sklearn vectorizer
        model:       sklearn classifier
        downsample:  if True, use only 20% of data (faster for SVM)
        test_size:   fraction for test split
        verbose:     print classification report

    Returns:
        dict with f1_macro, f1_hate_speech, report string
    """
    # Vectorize
    X_vec = vectorizer.fit_transform(X)

    # Downsample for speed
    if downsample:
        X_vec, _, Y, _ = train_test_split(X_vec, Y, test_size=0.8, random_state=42, stratify=Y)
        print(f"Downsampled to: {X_vec.shape[0]} samples")


    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vec, Y, test_size=test_size, random_state=42, stratify=Y
    )

    # Train
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Evaluate
    report = classification_report(Y_test, Y_pred, target_names=['not Hate Speech', 'Hate Speech'])
    f1_macro = f1_score(Y_test, Y_pred, average='macro')
    f1_hate_speech = f1_score(Y_test, Y_pred, pos_label=1)

    if verbose:
        print(report)

    return {
        'f1_macro': round(f1_macro, 4),
        'f1_hate_speech': round(f1_hate_speech, 4),
        'report': report
    }
