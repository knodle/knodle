import logging
import os

from torch import Tensor
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.data.download import MinioConnector
from knodle.model.logistic_regression_model import (
    LogisticRegressionModel,
)

from examples.ImdbDataset.utils import init_logger
from examples.utils import read_train_dev_test
from examples.trainer.preprocessing import get_tfidf_features
from knodle.trainer import TrainerConfig
from knodle.trainer.trainer import BaseTrainer

logger = logging.getLogger(__name__)

OUTPUT_CLASSES = 2
RANDOM_STATE = 123
TARGET_PATH = 'data/imdb'
MAX_FEATURES = 40000


def train_simple_ds_model():
    init_logger()
    if not not os.path.exists('data/imdb/mapping_rules_labels_t.lib'):
        minio_connect = MinioConnector()
        minio_connect.download_dir("datasets/imdb/processed/", TARGET_PATH)

    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, imdb_dataset, \
    mapping_rules_labels_t = \
        read_train_dev_test(
            TARGET_PATH)
    logger.info("Train knn tfidf similarity model")

    X_train = train_df.reviews_preprocessed
    X_dev = dev_df.reviews_preprocessed
    X_test = test_df.reviews_preprocessed

    tfidf_values = get_tfidf_features(
        imdb_dataset.reviews_preprocessed.values, path_to_cache="tutorials/ImdbDataset/tfidf.lib",
        max_features=MAX_FEATURES
    )

    train_dataset = TensorDataset(Tensor(tfidf_values[X_train.index].toarray()))
    dev_dataset = TensorDataset(Tensor(tfidf_values[X_dev.index].toarray()))

    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    custom_model_config = TrainerConfig(
        model=model, epochs=35, optimizer_=SGD(model.parameters(), lr=0.1)
    )

    trainer = BaseTrainer(
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

    clf_report = trainer.test(test_tfidf, y_test)
    print(clf_report)


if __name__ == "__main__":
    train_simple_ds_model()
