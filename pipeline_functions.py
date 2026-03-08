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

def get_model(name='svm'):
    """
    Returns a sklearn classifier.

    Args:
        name: 'svm', 'logreg', or 'naivebayes'

    Returns:
        classifier instance
    """
    if name == 'svm':
        return SVC(class_weight='balanced', random_state=42)
    elif name == 'logreg':
        return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
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


# ─────────────────────────────────────────────
# 5. RUN ALL EXPERIMENTS
# ─────────────────────────────────────────────

EXPERIMENTS = [
    {
        'id': 'E1',
        'name': 'Baseline',
        'preprocessing': 'stem+stop',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'svm',
    },
    {
        'id': 'E2',
        'name': 'No Preprocessing',
        'preprocessing': 'none',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'svm',
    },
    {
        'id': 'E3',
        'name': 'Stopwords only (no Stemming)',
        'preprocessing': 'stop',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'svm',
    },
    {
        'id': 'E4',
        'name': 'Stemming only (no Stopwords)',
        'preprocessing': 'stem',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'svm',
    },
    {
        'id': 'E5',
        'name': 'Bag of Words',
        'preprocessing': 'stem+stop',
        'vectorizer': 'bow',
        'max_features': 1000,
        'model': 'svm',
    },
    {
        'id': 'E6',
        'name': 'TF-IDF 5000 features',
        'preprocessing': 'stem+stop',
        'vectorizer': 'tfidf',
        'max_features': 5000,
        'model': 'svm',
    },
    {
        'id': 'E7',
        'name': 'Logistic Regression',
        'preprocessing': 'stem+stop',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'logreg',
    },
    {
        'id': 'E8',
        'name': 'Naive Bayes',
        'preprocessing': 'stem+stop',
        'vectorizer': 'tfidf',
        'max_features': 1000,
        'model': 'naivebayes',
    },
]

PREPROCESS_CONFIGS = {
    'none':      dict(remove_stopwords=False, remove_numbers=False, do_stem=False),
    'stop':      dict(remove_stopwords=True,  remove_numbers=True,  do_stem=False),
    'stem':      dict(remove_stopwords=False, remove_numbers=True,  do_stem=True),
    'stem+stop': dict(remove_stopwords=True,  remove_numbers=True,  do_stem=True),
}


def run_all_experiments(csv_path='data/train.csv', downsample=False, verbose=False):
    """
    Runs all 8 experiments and returns a results DataFrame.

    Args:
        csv_path:   path to train.csv
        downsample: use 20% of data for speed
        verbose:    print each classification report

    Returns:
        pd.DataFrame with results
    """
    results = []

    # Cache preprocessed versions to avoid redundant processing
    data_cache = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Running {exp['id']}: {exp['name']}")
        print(f"{'='*50}")

        prep_key = exp['preprocessing']

        # Load/preprocess data (use cache if already done)
        if prep_key not in data_cache:
            cfg = PREPROCESS_CONFIGS[prep_key]
            fn = lambda text, cfg=cfg: preprocess(text, **cfg)
            X, Y = load_and_preprocess(
                csv_path=csv_path,
                preprocess_fn=fn,
                cache_path=f'cache_{prep_key}',
                reprocess=False
            )
            data_cache[prep_key] = (X, Y)
        else:
            X, Y = data_cache[prep_key]
            print(f"Using cached data for preprocessing: {prep_key}")

        # Vectorizer & model
        vectorizer = get_vectorizer(exp['vectorizer'], exp['max_features'])
        model = get_model(exp['model'])

        # Run
        metrics = run_experiment(X, Y, vectorizer, model, downsample=downsample, verbose=verbose)

        results.append({
            'ID': exp['id'],
            'Name': exp['name'],
            'Preprocessing': exp['preprocessing'],
            'Vectorizer': f"{exp['vectorizer'].upper()} ({exp['max_features']})",
            'Model': exp['model'],
            'F1 Macro': metrics['f1_macro'],
            'F1 Hate Speech': metrics['f1_hate_speech'],
        })

        print(f"→ F1 Macro: {metrics['f1_macro']} | F1 Hate Speech: {metrics['f1_hate_speech']}")

    df_results = pd.DataFrame(results)
    return df_results