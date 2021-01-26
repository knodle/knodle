import logging

import os
import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.optim import AdamW, SGD
from tqdm import tqdm

from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer.knn_tfidf_similarities.knn_tfidf_similarity import (
    KnnTfidfSimilarity,
)
from knodle.trainer import TrainerConfig
from torch.utils.tensorboard import SummaryWriter
import wandb


logger = logging.getLogger(__name__)

OUTPUT_CLASSES = 2
DEV_SIZE = 1000
TEST_SIZE = 1000
RANDOM_STATE = 123
writer = SummaryWriter()


def train_knn_model():
    logger.info("Train knn tfidf similarity model")

    imdb_dataset, rule_matches_z, mapping_rules_labels_t = read_evaluation_data()

    review_series = imdb_dataset.reviews_preprocessed
    label_ids = imdb_dataset.label_id

    rest, dev = train_test_split(
        imdb_dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE
    )

    train, test = train_test_split(rest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = train.reviews_preprocessed
    X_dev = dev.reviews_preprocessed
    X_test = test.reviews_preprocessed

    y_train = train.label_id
    y_dev = dev.label_id
    y_test = test.label_id

    max_features = 40000

    tfidf_values = create_tfidf_values(
        imdb_dataset.reviews_preprocessed.values, True, max_features
    )

    train_rule_matches_z = rule_matches_z[X_train.index]
    train_tfidf_sparse = tfidf_values[X_train.index]
    train_tfidf = Tensor(tfidf_values[X_train.index].toarray())
    train_dataset = TensorDataset(train_tfidf)

    dev_rule_matches_z = rule_matches_z[X_dev.index]
    dev_tfidf_sparse = tfidf_values[X_dev.index]
    dev_tfidf = Tensor(tfidf_values[X_dev.index].toarray())

    dev_dataset = TensorDataset(dev_tfidf)

    for current_k in range(1, 100, 1):
        print(current_k)
        model = LogisticRegressionModel(tfidf_values.shape[1], 2)

        custom_model_config = TrainerConfig(
            model=model, epochs=35, optimizer_=SGD(model.parameters(), lr=0.1)
        )

        trainer = KnnTfidfSimilarity(
            model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            tfidf_values=train_tfidf_sparse,
            model_input_x=train_dataset,
            rule_matches_z=train_rule_matches_z,
            dev_model_input_x=dev_dataset,
            dev_rule_matches_z=dev_rule_matches_z,
            k=current_k,
            cache_denoised_matches=True,
            caching_prefix="imdb_",
            trainer_config=custom_model_config,
        )
        experiment_name = wandb.util.generate_id()

        trainer.trainer_config.k = current_k

        wandb.init(
            project="knodle",
            group=experiment_name,
            config=trainer.trainer_config)

        trainer.train()

        tfidf_values_sparse = Tensor(tfidf_values[X_test.index].toarray())
        tfidf_values_sparse = tfidf_values_sparse.to(custom_model_config.device)

        test_tfidf = TensorDataset(tfidf_values_sparse)

        y_test = Tensor(imdb_dataset.loc[X_test.index, "label_id"].values)
        y_test = y_test.to(custom_model_config.device)
        y_test = TensorDataset(y_test)

        clf_report = trainer.test(test_features=test_tfidf, test_labels=y_test)

        wandb.log({
            "test_accuracy": clf_report["accuracy"],
            "f1_weighted": clf_report.get('weighted avg').get('f1-score'),
            "recall_weighted": clf_report.get('weighted avg').get('precision'),
            "precision_weighted": clf_report.get('weighted avg').get('recall'),

        },)

        writer.add_hparams(
            {
                "type": "knn",
                "epochs": trainer.trainer_config.epochs,
                "optimizer": str(trainer.trainer_config.optimizer),
                "learning_rate": trainer.trainer_config.optimizer.defaults["lr"],
                "k": trainer.trainer_config.k,
            },
            {
                "test_accuracy": clf_report["accuracy"],
                "f1_weighted": clf_report.get('weighted avg').get('f1-score'),
                "recall_weighted": clf_report.get('weighted avg').get('precision'),
                "precision_weighted": clf_report.get('weighted avg').get('recall'),

            },
        )

        wandb.finish()


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
