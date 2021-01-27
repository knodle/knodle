import argparse
import os
import sys
import torch
from joblib import load
import pandas as pd
from torchtext.vocab import GloVe

from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.model.logistic_regression.logisitc_regression_with_emb_layer import (
    LogisticRegressionModel,
)
from knodle.trainer.config.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.config.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeigh
from tutorials.crossweigh_weighing_example import utils
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from tutorials.crossweigh_weighing_example.utils import encode_samples

NUM_CLASSES = 2
MAXLEN = 50


# CLASS_WEIGHTS = torch.FloatTensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
#                                    2.0, 2.0, 2.0, 2.0, 2.0])


def train_crossweigh(
        path_t: str,
        path_train_samples: str,
        path_z: str,
        path_word_emb_file: str,
        path_dev_features_labels: str = None,
        sample_weights: str = None,
) -> None:
    """
    Training the model with CrossWeigh model denoising
    :param path_train_samples: path to DataFrame with training data
    :param path_z: path to binary matrix that contains info about rules matched in samples (samples x rules)
    :param path_t: path to binary matrix that contains info about which rule corresponds to which label (rule x labels)
    :param path_dev_features_labels: path to DataFrame with development data (1st column - samples, 2nd column - labels)
    :param path_word_emb_file: path to file with pretrained embeddings
    """
    # embedding_glove = GloVe(name='6B', dim=300)

    word2id, word_embedding_matrix = utils.vocab_and_vectors(path_word_emb_file)

    input_data = pd.read_csv(path_train_samples)
    rule_matches_z = load(path_z)
    rule_assignments_t = load(path_t)

    if path_dev_features_labels is not None:
        # the test set is given for testing the function
        train_input_x = utils.get_data_features(
            input_data, word2id, samples_column=2, maxlen=MAXLEN
        )
        input_dev_data = pd.read_csv(path_dev_features_labels)
        dev_input_x, dev_labels = utils.get_data_features(
            input_dev_data, word2id, samples_column=1, labels_column=2, maxlen=MAXLEN
        )

    else:
        # the dataset should be splitted into train and test set
        data_series = input_data.reviews_preprocessed
        label_ids = input_data.label_id
        train_data_spl, dev_data_spl, _, dev_labels = train_test_split(
            data_series, label_ids, test_size=0.2, random_state=0
        )

        train_input_x = TensorDataset(
            torch.LongTensor(
                encode_samples(list(train_data_spl.values), word2id, MAXLEN)
            )
        )
        train_rule_matches_z = rule_matches_z[train_data_spl.index]

        dev_input_x = torch.LongTensor(
            encode_samples(list(dev_data_spl.values), word2id, MAXLEN)
        )
        dev_labels = torch.LongTensor(
            input_data.loc[dev_data_spl.index, "label_id"].values
        )
        dev_input_x_dataset = TensorDataset(dev_input_x)
        dev_labels_dataset = TensorDataset(dev_labels)

    dev_features_and_labels = TensorDataset(dev_input_x, dev_labels)

    model = LogisticRegressionModel(
        MAXLEN,
        word_embedding_matrix.shape[0],
        word_embedding_matrix.shape[1],
        word_embedding_matrix,
        NUM_CLASSES,
    )

    custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
        model=model,
        # class_weights=CLASS_WEIGHTS,
        lr=0.8,
        output_classes=NUM_CLASSES,
        negative_samples=False,
    )
    custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
        model=model,
        # class_weights=CLASS_WEIGHTS,
        lr=0.01,
        output_classes=NUM_CLASSES,
        epochs=35,
        negative_samples=False,
    )

    trainer = CrossWeigh(
        model=model,
        rule_assignments_t=rule_assignments_t,
        inputs_x=train_input_x,
        rule_matches_z=train_rule_matches_z,
        dev_features_labels=dev_features_and_labels,
        weights=sample_weights,
        denoising_config=custom_crossweigh_denoising_config,
        trainer_config=custom_crossweigh_trainer_config,
    )
    trainer.train()
    trainer.test(test_features=dev_input_x_dataset, test_labels=dev_labels_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--rule_assignments_t", help="")
    parser.add_argument("--path_train_samples", help="")
    parser.add_argument("--rule_matches_z", help="")
    parser.add_argument("--word_embeddings", help="")
    parser.add_argument("--dev_features_labels", help="")
    parser.add_argument(
        "--sample_weights", help="If there are pretrained samples sample_weights"
    )

    args = parser.parse_args()

    train_crossweigh(
        args.rule_assignments_t,
        args.path_train_samples,
        args.rule_matches_z,
        args.word_embeddings,
        args.dev_features_labels,
        args.sample_weights,
    )
