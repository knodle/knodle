import logging
import os

from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

from knodle.data.download import MinioConnector
from knodle.model.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer.knn_denoising.config import KNNConfig
from knodle.trainer.knn_denoising.knn_denoising import (
    KnnDenoisingTrainer,
)
from tutorials.ImdbDataset.utils import read_train_dev_test, create_tfidf_values, init_logger

logger = logging.getLogger(__name__)

OUTPUT_CLASSES = 2
RANDOM_STATE = 123
writer = SummaryWriter()
TARGET_PATH = 'data/imdb'
MAX_FEATURES = 40000


def train_knn_model():
    init_logger()
    if not not os.path.exists('data/imdb/mapping_rules_labels_t.lib'):
        minio_connect = MinioConnector()
        minio_connect.download_dir("datasets/imdb/processed", TARGET_PATH)

    train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, imdb_dataset, \
    mapping_rules_labels_t = \
        read_train_dev_test(
            TARGET_PATH)
    logger.info("Train knn tfidf similarity model")

    X_train = train_df.reviews_preprocessed
    X_dev = dev_df.reviews_preprocessed
    X_test = test_df.reviews_preprocessed

    tfidf_values = create_tfidf_values(
        imdb_dataset.reviews_preprocessed.values, True, MAX_FEATURES
    )

    train_dataset = TensorDataset(Tensor(tfidf_values[X_train.index].toarray()))
    dev_dataset = TensorDataset(Tensor(tfidf_values[X_dev.index].toarray()))


    model = LogisticRegressionModel(tfidf_values.shape[1], 2)
    for k in [1, 2, 4, 8, 15]:
        print("K is: {}".format(k))
        custom_model_config = KNNConfig(
            model=model, epochs=35, optimizer_=AdamW(model.parameters(), lr=0.01),
            k=k
        )

        tfidf_values_sparse = Tensor(tfidf_values[X_test.index].toarray())
        tfidf_values_sparse = tfidf_values_sparse.to(custom_model_config.device)

        test_tfidf = TensorDataset(tfidf_values_sparse)

        y_dev = Tensor(imdb_dataset.loc[X_dev.index, "label_id"].values)
        y_dev = y_dev.to(custom_model_config.device)
        y_dev = TensorDataset(y_dev)

        y_test = Tensor(imdb_dataset.loc[X_test.index, "label_id"].values)
        y_test = y_test.to(custom_model_config.device)
        y_test = TensorDataset(y_test)

        trainer = KnnDenoisingTrainer(
            model=model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            model_input_x=train_dataset,
            rule_matches_z=train_rule_matches_z,
            dev_model_input_x=dev_dataset,
            dev_gold_labels_y=y_dev,
            trainer_config=custom_model_config,
        )

        trainer.train()

        clf_report, _ = trainer.test(test_features=test_tfidf, test_labels=y_test)
        print("-------------------------")
        print(f"k == {k}")
        print(clf_report)



if __name__ == "__main__":
    train_knn_model()
