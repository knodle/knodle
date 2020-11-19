from sklearn.feature_extraction.text import TfidfVectorizer

from knodle.model import LogisticRegressionModel
from joblib import load
import pandas as pd
from torch import Tensor

from knodle.trainer.SimpleDsModelTrainer.SimpleDsModelTrainer import (
    SimpleDsModelTrainer,
)


def train_simple_ds_model():
    print("Train simple ds model")
    imdb_dataset, applied_lfs = read_evaluation_data()
    tfidf_values = create_tfidf_values(imdb_dataset.reviews_preprocessed.values)
    tfidf_tensor = Tensor(tfidf_values.toarray())
    model = LogisticRegressionModel(tfidf_values.shape[1], 2)
    trainer = SimpleDsModelTrainer(model)
    trainer.train(inputs=tfidf_tensor, applied_labeling_functions=applied_lfs, epochs=5)


def read_evaluation_data():
    imdb_dataset = pd.read_csv("tutorials/ImdbDataset/imdb_data_pp.csv")
    applied_lfs = load("tutorials/ImdbDataset/lfs.joblib")
    return imdb_dataset, applied_lfs


def create_tfidf_values(text_data: [str]):
    vectorizer = TfidfVectorizer()
    transformed_data = vectorizer.fit_transform(text_data)
    return transformed_data


if __name__ == "__main__":
    train_simple_ds_model()
