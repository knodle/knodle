import os

from sklearn.feature_extraction.text import TfidfVectorizer

from knodle.model import LogisticRegressionModel
from joblib import load, dump
import pandas as pd
from torch import Tensor

from knodle.trainer import KnnDenoising
import logging

logger = logging.getLogger(__name__)


def train_knn_model():
    logger.info("Train knn model")
    imdb_dataset, applied_lfs = read_evaluation_data()
    tfidf_values = create_tfidf_values(
        imdb_dataset.reviews_preprocessed.values
    ).toarray()

    tfidf_tensor = Tensor(tfidf_values)
    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    trainer = KnnDenoising(model)
    trainer.train(
        inputs=tfidf_tensor,
        applied_labeling_functions=applied_lfs,
        tfidf_values=tfidf_values,
        epochs=2,
        k=2,
    )


def read_evaluation_data():
    imdb_dataset = pd.read_csv("tutorials/ImdbDataset/imdb_data_preprocessed.csv")
    applied_lfs = load("tutorials/ImdbDataset/applied_lfs.lib")
    return imdb_dataset, applied_lfs


def create_tfidf_values(text_data: [str]):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib"):
        return load("tutorials/ImdbDataset/tfidf.lib")
    else:
        vectorizer = TfidfVectorizer()
        transformed_data = vectorizer.fit_transform(text_data)
        dump(transformed_data, "tutorials/ImdbDataset/tfidf.lib")
        return transformed_data


if __name__ == "__main__":
    train_knn_model()
