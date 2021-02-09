
import logging

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer.knn_tfidf_similarities.knn_tfidf_similarity import (
    KnnTfidfSimilarity,
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 42
MAXLEN = 50
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])


def train_knn_model(
        path_t: str,
        path_train_samples: str,
        path_z: str,
        path_sample_weights: str = None,
        path_dev_features_labels: str = None,
        path_test_features_labels: str = None
) -> None:

    logger.info("Train knn tfidf similarity model")
    train_data = pd.read_csv(path_train_samples)
    rule_matches_z = load(path_z)
    mapping_rules_labels_t = load(path_t)
    dev_data = pd.read_csv(path_dev_features_labels)
    test_data = pd.read_csv(path_test_features_labels)

    train_data, rule_matches_z, mapping_rules_labels_t, dev_data, test_data = read_evaluation_data()

    samples_series = train_data.samples

    # X_train, X_test, y_train, y_test = train_test_split(
    #     review_series, label_ids, test_size=0.2, random_state=0
    # )

    train_tfidf_sparse, test_tfidf_sparse = create_tfidf_values(train_data.samples.values, test_data.samples.values)
    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset = TensorDataset(train_tfidf)

    test_tfidf_values = TensorDataset(Tensor(test_tfidf_sparse.toarray()))
    test_labels = Tensor(test_data.enc_labels.values)

    model = LogisticRegressionModel(train_tfidf.shape[1], 41)

    from knodle.trainer import TrainerConfig
    custom_config = TrainerConfig(model, lr=0.8, epochs=20)

    trainer = KnnTfidfSimilarity(
        model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        tfidf_values=train_tfidf_sparse,
        model_input_x=train_dataset,
        rule_matches_z=rule_matches_z,
        k=1,
        trainer_config=custom_config
    )

    trainer.train()

    trainer.test(features_dataset=test_tfidf_values, labels=test_labels)


def read_evaluation_data():
    train_data = pd.read_csv(
        "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_banned_16_24_39_with_norel/train_samples.csv")
    rule_matches_z = load(
        "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_banned_16_24_39_with_norel/z_matrix")
    mapping_rules_labels_t = load(
        "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_banned_16_24_39_with_norel/t_matrix")

    dev_data = pd.read_csv(
        "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_banned_16_24_39_with_norel/dev_samples.csv")
    test_data = pd.read_csv(
        "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_banned_16_24_39_with_norel/test_samples.csv")

    return train_data, rule_matches_z, mapping_rules_labels_t, dev_data, test_data


def create_tfidf_values(train_data: [str], test_data: [str]):
    # if os.path.exists("/Users/asedova/PycharmProjects/knodle/tutorials/ImdbDataset/data/tfidf.lib"):
    #     cached_data = load("/Users/asedova/PycharmProjects/knodle/tutorials/ImdbDataset/data/tfidf.lib")
    #     if cached_data.shape == text_data.shape:
    #         return cached_data

    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(train_data)
    test_transformed_data = vectorizer.transform(test_data)
    # dump(transformed_data, "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_with_added_arg_types/tfidf.lib")
    return train_transformed_data, test_transformed_data


if __name__ == "__main__":
    train_knn_model()