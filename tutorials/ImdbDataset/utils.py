import logging
import os
import sys

from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_values(
        text_data: [str], force_create_new: bool = False, max_features: int = None
):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib") and not force_create_new:
        cached_data = load("tutorials/ImdbDataset/tfidf.lib")
        if cached_data.shape == text_data.shape:
            return cached_data

    vectorizer = TfidfVectorizer(min_df=2, max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, "tutorials/ImdbDataset/tfidf.lib")
    return transformed_data


def init_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
