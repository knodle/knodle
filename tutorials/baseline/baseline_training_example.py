import logging
import os

import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer

logger = logging.getLogger(__name__)

OUTPUT_CLASSES = 2
DEV_SIZE = 1000
TEST_SIZE = 1000
RANDOM_STATE = 123


def train_simple_ds_model():
    logger.info("Train simple ds model")
    imdb_dataset, rule_matches_z, mapping_rules_labels_t = read_evaluation_data()

    rest, dev = train_test_split(
        imdb_dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE
    )

    train, test = train_test_split(rest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = train.reviews_preprocessed
    X_test = test.reviews_preprocessed

    max_features = 40000

    tfidf_values = create_tfidf_values(
        imdb_dataset.reviews_preprocessed.values, True, max_features
    )

    train_rule_matches_z = rule_matches_z[X_train.index]
    train_tfidf = Tensor(tfidf_values[X_train.index].toarray())

    train_dataset = TensorDataset(train_tfidf)

    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    custom_model_config = TrainerConfig(
        model=model, epochs=1, optimizer_=SGD(model.parameters(), lr=0.1)
    )
    trainer = SimpleDsModelTrainer(
        model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=train_dataset,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_model_config,
    )
    trainer.train()

    tfidf_values_sparse = Tensor(tfidf_values[X_test.index].toarray())
    tfidf_values_sparse = tfidf_values_sparse.to(custom_model_config.device)

    test_tfidf = TensorDataset(tfidf_values_sparse)

    y_test = Tensor(imdb_dataset.loc[X_test.index, "label_id"].values)
    y_test = y_test.to(custom_model_config.device)
    y_test = TensorDataset(y_test)

    clf_report = trainer.test(test_features=test_tfidf, test_labels=y_test)
    print(clf_report)


def read_evaluation_data():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "ImdbDataset"
    )
    imdb_dataset = pd.read_csv(os.path.join(data_path, "imdb_data_preprocessed.csv"))
    rule_matches_z = load(os.path.join(data_path, "rule_matches.lib"))
    mapping_rules_labels_t = load(os.path.join(data_path, "mapping_rules_labels.lib"))
    return imdb_dataset, rule_matches_z, mapping_rules_labels_t


def create_tfidf_values(
    text_data: [str], force_create_new: bool = False, max_features: int = None
):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib") and not force_create_new:
        cached_data = load("tutorials/ImdbDataset/tfidf_{}.lib".format(max_features))
        if cached_data.shape == text_data.shape:
            return cached_data

    vectorizer = TfidfVectorizer(min_df=2, max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, "tutorials/ImdbDataset/tfidf_{}.lib".format(max_features))
    return transformed_data


if __name__ == "__main__":
    train_simple_ds_model()
