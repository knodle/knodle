import os

from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW

from knodle.model import LogisticRegressionModel
from joblib import load, dump
import pandas as pd
from torch import Tensor
from sklearn.model_selection import train_test_split

import logging

from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer

logger = logging.getLogger(__name__)


def train_simple_ds_model():
    logger.info("Train simple ds model")
    imdb_dataset, rule_matches = read_evaluation_data()

    review_series = imdb_dataset.reviews_preprocessed
    label_ids = imdb_dataset.label_id

    X_train, X_test, y_train, y_test = train_test_split(
        review_series, label_ids, test_size=0.2, random_state=0
    )

    tfidf_values = create_tfidf_values(imdb_dataset.reviews_preprocessed.values)

    train_rule_matches = rule_matches[X_train.index]
    train = Tensor(tfidf_values[X_train.index].toarray())
    test = Tensor(tfidf_values[X_test.index].toarray())
    y_test = Tensor(imdb_dataset.loc[X_test.index, "label_id"].values)

    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    custom_model_config = TrainerConfig(
        model=model, optimizer_=AdamW(model.parameters(), lr=0.01)
    )

    trainer = SimpleDsModelTrainer(model, trainer_config=custom_model_config)
    trainer.train(inputs=train, rule_matches=train_rule_matches, epochs=20)

    trainer.test(test_features=test, test_labels=Tensor(y_test))


def read_evaluation_data():
    imdb_dataset = pd.read_csv("tutorials/ImdbDataset/imdb_data_preprocessed.csv")
    applied_lfs = load("tutorials/ImdbDataset/applied_lfs.lib")
    return imdb_dataset, applied_lfs


def create_tfidf_values(text_data: [str]):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib"):
        cached_data = load("tutorials/ImdbDataset/tfidf.lib")
        if cached_data.shape == text_data.shape:
            return cached_data

    vectorizer = TfidfVectorizer()
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, "tutorials/ImdbDataset/tfidf.lib")
    return transformed_data


if __name__ == "__main__":
    train_simple_ds_model()
