import logging

import os
import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import NoDenoisingTrainer

logger = logging.getLogger(__name__)


def train_simple_ds_model():
    logger.info("Train simple ds model")
    imdb_dataset, rule_matches_z, mapping_rules_labels_t = read_evaluation_data()

    review_series = imdb_dataset.reviews_preprocessed
    label_ids = imdb_dataset.label_id

    X_train, X_test, y_train, y_test = train_test_split(
        review_series, label_ids, test_size=0.2, random_state=0
    )

    tfidf_values = create_tfidf_values(imdb_dataset.reviews_preprocessed.values)

    train_rule_matches_z = rule_matches_z[X_train.index]
    train_tfidf = Tensor(tfidf_values[X_train.index].toarray())
    test_tfidf = TensorDataset(Tensor(tfidf_values[X_test.index].toarray()))
    y_test = TensorDataset(Tensor(imdb_dataset.loc[X_test.index, "label_id"].values))

    train_dataset = TensorDataset(train_tfidf)

    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    custom_model_config = TrainerConfig(
        model=model, optimizer_=AdamW(model.parameters(), lr=0.01)
    )

    trainer = NoDenoisingTrainer(
        model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=train_dataset,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_model_config,
    )
    trainer.train()

    trainer.test(test_features=test_tfidf, test_labels=y_test)


def read_evaluation_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ImdbDataset")
    imdb_dataset = pd.read_csv(os.path.join(data_path, "imdb_data_preprocessed.csv"))
    rule_matches_z = load(os.path.join(data_path, "rule_matches.lib"))
    mapping_rules_labels_t = load(os.path.join(data_path, "mapping_rules_labels.lib"))
    return imdb_dataset, rule_matches_z, mapping_rules_labels_t


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
