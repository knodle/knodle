import argparse
import os
import sys
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import scipy
import torch
from joblib import load
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset

from knodle.evaluation.tacred_metrics import score
from tutorials.crossweigh_weighing_example.utils import vocab_and_vectors
from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeighTrainer

from knodle.trainer.config import TrainerConfig
from knodle.trainer.crossweigh_weighing.config import CrossWeighDenoisingConfig

NUM_CLASSES = 42
MAXLEN = 50
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])


def train_crossweigh(
        path_t: str,
        path_train_samples: str,
        path_z: str,
        path_labels: str,
        path_emb: str,
        path_sample_weights: str = None,
        path_dev_features_labels: str = None,
        path_test_features_labels: str = None
) -> None:
    """
    Training the model with CrossWeigh model denoising
    :param path_sample_weights:
    :param path_test_features_labels:
    :param path_emb: path to file with pretrained embeddings
    :param path_train_samples: path to DataFrame with training data
    :param path_z: path to binary matrix that contains info about rules matched in samples (samples x rules)
    :param path_labels: path to file with labels
    :param path_t: path to binary matrix that contains info about which rule corresponds to which label (rule x labels)
    :param path_dev_features_labels: path to DataFrame with development data (1st column - samples, 2nd column - labels)
    """

    ids2labels = read_labels_from_file(path_labels, "no_relation")
    word2id, word_embedding_matrix = vocab_and_vectors(path_emb)

    rule_matches_z = load(path_z)
    rule_matches_z = rule_matches_z.toarray() if scipy.sparse.issparse(rule_matches_z) else rule_matches_z
    rule_assignments_t = load(path_t)
    train_input_x = get_train_features(path_train_samples, word2id, samples_column_num=1)

    dev_dataset, dev_labels = get_dev_data(path_dev_features_labels, word2id, samples_column_num=1, labels_column_num=4)
    test_dataset, test_labels = get_dev_data(path_test_features_labels, word2id, samples_column_num=1,
                                             labels_column_num=4)

    path_to_weights = os.path.join(path_sample_weights)
    os.makedirs(path_to_weights, exist_ok=True)

    model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                              word_embedding_matrix.shape[1],
                              word_embedding_matrix,
                              NUM_CLASSES)

    custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
        model=model,
        partitions=2,
        class_weights=CLASS_WEIGHTS,
        crossweigh_folds=10,
        epochs=2,
        weight_reducing_rate=0.3,
        samples_start_weights=3.0,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.8),
        output_classes=NUM_CLASSES
    )

    custom_crossweigh_trainer_config = TrainerConfig(
        model=model,
        class_weights=CLASS_WEIGHTS,
        output_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters(), lr=2.0),
        epochs=3
    )

    trainer = CrossWeighTrainer(
        model=model,
        rule_assignments_t=rule_assignments_t,
        inputs_x=train_input_x,
        rule_matches_z=rule_matches_z,
        dev_features=dev_dataset,
        dev_labels=dev_labels,
        evaluation_method="tacred",
        ids2labels=ids2labels,
        path_to_weights=path_to_weights,
        denoising_config=custom_crossweigh_denoising_config,
        trainer_config=custom_crossweigh_trainer_config,
        run_classifier=True,
        use_weights=True
    )

    trainer.train()
    print("Testing on the test dataset....")
    metrics = test(model, trainer, test_dataset, test_labels, labels2ids)
    print(metrics)


def get_train_features(
        path_train_data: str, word2id: dict, samples_column_num: int
) -> TensorDataset:
    """
    This function reads the input data saved as a DataFrame and encode sentences with words ids.
    :param path_train_data: path to .csv file with input data
    :param word2id: dictionary of words to their ids that corresponds to pretrained embeddings
    :param samples_column_num: number of a column in input DataFrame where samples are stored
    :return:
    """
    input_data = pd.read_csv(path_train_data)
    enc_input_samples = encode_samples(list(input_data.iloc[:, samples_column_num]), word2id, MAXLEN)
    inputs_x_tensor = torch.LongTensor(enc_input_samples)
    inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)
    return inputs_x_dataset


def get_dev_data(
        path_dev_feature_labels: str, word2id: dict, samples_column_num: int, labels_column_num: int
) -> Tuple[TensorDataset, LongTensor]:
    """ Read dev data with gold labels and turn it into TensorDataset(features, labels)"""
    dev_data = pd.read_csv(path_dev_feature_labels)
    enc_dev_samples = encode_samples(list(dev_data.iloc[:, samples_column_num]), word2id, MAXLEN)
    dev_samples_dataset = torch.utils.data.TensorDataset(torch.LongTensor(enc_dev_samples))
    dev_labels_tensor = torch.LongTensor(list(dev_data.iloc[:, labels_column_num]))
    return dev_samples_dataset, dev_labels_tensor


def read_labels_from_file(path_labels: str, negative_label: str) -> dict:
    """ Reads the labels from the file and encode them with ids """
    ids2relation = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            ids2relation[int(relation_enc)] = relation

    # add no_match label
    if negative_label:
        ids2relation[max(list(ids2relation.keys())) + 1] = negative_label

    return ids2relation


def encode_samples(raw_samples: list, word2id: dict, maxlen: int) -> list:
    """ This function turns raw text samples into encoded ones using the given word2id dict """
    enc_input_samples = []
    for sample in raw_samples:
        enc_tokens = [word2id.get(token, 1) for token in sample.lstrip().split(" ")]
        enc_input_samples.append(
            np.asarray(add_padding(enc_tokens, maxlen), dtype="float32")
        )
    return enc_input_samples


def add_padding(tokens: list, maxlen: int) -> list:
    """ Provide padding of the encoded tokens to the maxlen; if length of tokens > maxlen, reduce it to maxlen """
    padded_tokens = [0] * maxlen
    for token in range(0, min(len(tokens), maxlen)):
        padded_tokens[token] = tokens[token]
    return padded_tokens


def test(model, trainer, test_features: TensorDataset, test_labels: Tensor, labels2ids: Dict) -> Dict:
    feature_labels_dataset = TensorDataset(test_features.tensors[0], test_labels)
    feature_labels_dataloader = trainer._make_dataloader(feature_labels_dataset)

    model.eval()
    all_predictions, all_labels = torch.Tensor(), torch.Tensor()
    for features, labels in feature_labels_dataloader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        all_predictions = torch.cat([all_predictions, predicted])
        all_labels = torch.cat([all_labels, labels])

    predictions_idx, test_labels_idx = (all_predictions.detach().type(torch.IntTensor).tolist(),
                                        all_labels.detach().type(torch.IntTensor).tolist())

    idx2labels = dict([(value, key) for key, value in labels2ids.items()])

    predictions = [idx2labels[p] for p in predictions_idx]
    test_labels = [idx2labels[p] for p in test_labels_idx]

    return score(test_labels, predictions, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--rule_assignments_t", help="")
    parser.add_argument("--path_train_samples", help="")
    parser.add_argument("--rule_matches_z", help="")
    parser.add_argument("--path_labels", help="")
    parser.add_argument("--word_embeddings", help="")
    parser.add_argument("--sample_weights", help="")
    parser.add_argument("--dev_features_labels", help="")
    parser.add_argument("--test_features_labels", help="")

    args = parser.parse_args()

    train_crossweigh(args.rule_assignments_t,
                     args.path_train_samples,
                     args.rule_matches_z,
                     args.path_labels,
                     args.word_embeddings,
                     args.sample_weights,
                     args.dev_features_labels,
                     args.test_features_labels)
