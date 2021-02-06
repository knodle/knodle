import logging
import os
import sys

import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import SGD, AdamW
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer.knn_tfidf_similarities.knn_config import KNNConfig
from knodle.trainer.knn_tfidf_similarities.knn_tfidf_similarity import (
    KnnTfidfSimilarity,
)

logger = logging.getLogger(__name__)

OUTPUT_CLASSES = 2
# DEV_SIZE = 1000
# TEST_SIZE = 1000

DEV_SIZE = 5000
TEST_SIZE = 5000
RANDOM_STATE = 123
writer = SummaryWriter()


def train_knn_model():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    logger.info("Train knn tfidf similarity model")

    imdb_dataset, rule_matches_z, mapping_rules_labels_t = read_evaluation_data()

    rest, dev = train_test_split(
        imdb_dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE
    )

    train, test = train_test_split(rest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = train.reviews_preprocessed
    X_dev = dev.reviews_preprocessed
    X_test = test.reviews_preprocessed

    max_features = 200

    tfidf_values = create_tfidf_values(
        imdb_dataset.reviews_preprocessed.values, True, max_features
    )

    train_rule_matches_z = rule_matches_z[X_train.index]
    train_tfidf_sparse = tfidf_values[X_train.index]
    train_tfidf = Tensor(tfidf_values[X_train.index].toarray())
    train_dataset = TensorDataset(train_tfidf)

    dev_rule_matches_z = rule_matches_z[X_dev.index]
    dev_tfidf = Tensor(tfidf_values[X_dev.index].toarray())

    dev_dataset = TensorDataset(dev_tfidf)

    model = LogisticRegressionModel(tfidf_values.shape[1], 2)
    for k in [2, 2, 4, 8, 15]:
        custom_model_config = KNNConfig(
            model=model, epochs=3, optimizer_=AdamW(model.parameters(), lr=0.01),
            k=k  # , caching_folder=os.path.join(os.getcwd(), "data/knn_caching")
        )

        trainer = KnnTfidfSimilarity(
            model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            model_input_x=train_dataset,
            rule_matches_z=train_rule_matches_z,
            dev_model_input_x=dev_dataset,
            dev_rule_matches_z=dev_rule_matches_z,
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
        print("-------------------------")
        print(f"k == {k}")
        print(clf_report)


def read_evaluation_data():
    imdb_dataset = pd.read_csv("tutorials/ImdbDataset/imdb_data_preprocessed.csv")
    rule_matches_z = load("tutorials/ImdbDataset/rule_matches.lib")
    mapping_rules_labels_t = load("tutorials/ImdbDataset/mapping_rules_labels.lib")
    return imdb_dataset, rule_matches_z, mapping_rules_labels_t


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


if __name__ == "__main__":
    train_knn_model()
