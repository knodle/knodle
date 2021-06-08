import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor, LongTensor
from torch.optim import Adam
from torch.utils.data import TensorDataset

from knodle.evaluation.other_class_metrics import score
from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.trainer.wscrossweigh.config import WSCrossWeighConfig
from knodle.trainer.wscrossweigh.wscrossweigh import WSCrossWeighTrainer
from examples.utils import read_train_dev_test, get_samples_list

NUM_CLASSES = 42
MAXLEN = 50
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])


def train_wscrossweigh(
        path_to_data: str,
        path_labels: str,
        path_emb: str,
        path_sample_weights: str = None,
) -> None:
    """ Training the BiLSTM model with WSCrossWeigh denoising algorithm """

    labels2ids = read_labels_from_file(path_labels, "no_relation")
    word2id, word_embedding_matrix = vocab_and_vectors(path_emb)

    df_train, df_dev, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels = \
        read_train_dev_test(path_to_data, if_dev_data=True)

    train_input_x = get_samples_features(df_train, word2id, samples_column_num=1)

    dev_dataset = get_samples_features(df_dev, word2id, samples_column_num=1)
    dev_labels_dataset = TensorDataset(LongTensor(list(df_dev.iloc[:, 4])))

    test_dataset = get_samples_features(df_test, word2id, samples_column_num=1)
    test_labels_dataset = TensorDataset(LongTensor(list(df_test.iloc[:, 4])))

    os.makedirs(path_sample_weights, exist_ok=True)

    parameters = {
        "lr": 1e-4,
        "cw_lr": 0.8,
        "epochs": 5,
        "cw_partitions": 2,
        "cw_folds": 5,
        "cw_epochs": 2,
        "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }

    model = BidirectionalLSTM(
        word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], word_embedding_matrix, NUM_CLASSES
    )

    custom_wscrossweigh_config = WSCrossWeighConfig(
        output_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS,
        filter_non_labelled=True,
        if_set_seed=True,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=Adam,
        lr=parameters.get("lr"),
        grad_clipping=5,
        partitions=parameters.get("cw_partitions"),
        folds=parameters.get("cw_folds"),
        weight_reducing_rate=parameters.get("weight_rr"),
        samples_start_weights=parameters.get("samples_start_weights")
    )

    trainer = WSCrossWeighTrainer(
        model=model,
        mapping_rules_labels_t=t_mapping_rules_labels,
        model_input_x=train_input_x,
        dev_model_input_x=dev_dataset,
        dev_gold_labels_y=dev_labels_dataset,
        rule_matches_z=z_train_rule_matches,
        trainer_config=custom_wscrossweigh_config,
        evaluation_method="tacred",
        dev_labels_ids=labels2ids,
        use_weights=True,
        run_classifier=True
    )
    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels_dataset)
    print(clf_report)


def get_samples_features(input_data: pd.DataFrame, word2id: dict, samples_column_num: int = None) -> TensorDataset:
    """ Encodes input samples with glove vectors and returns as a Dataset """
    enc_input_samples = encode_samples(get_samples_list(input_data, samples_column_num), word2id, MAXLEN)
    inputs_x_tensor = torch.LongTensor(enc_input_samples)
    inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)
    return inputs_x_dataset


def read_labels_from_file(path_labels: str, negative_label: str) -> dict:
    """ Reads the labels from the file and encode them with ids """
    relation2ids = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            relation2ids[relation] = int(relation_enc)

    # add no_match label
    if negative_label:
        relation2ids[negative_label] = max(list(relation2ids.values())) + 1

    return relation2ids


def encode_samples(raw_samples: list, word2id: dict, maxlen: int) -> list:
    """ This function turns raw text samples into encoded ones using the given word2id dict """
    enc_input_samples = []
    for sample in raw_samples:
        enc_tokens = [word2id.get(token, 1) for token in sample.lstrip().split(" ")]
        enc_input_samples.append(np.asarray(add_padding(enc_tokens, maxlen), dtype="float32"))
    return enc_input_samples


def add_padding(tokens: list, maxlen: int) -> list:
    """ Provide padding of the encoded tokens to the maxlen; if length of tokens > maxlen, reduce it to maxlen """
    padded_tokens = [0] * maxlen
    for token in range(0, min(len(tokens), maxlen)):
        padded_tokens[token] = tokens[token]
    return padded_tokens


def vocab_and_vectors(filename: str) -> (dict, np.ndarray):
    """
    Reads pretrained word embedding and builds 1) a matrix (words x embedding dim), 2) word to id dictionary
    :param filename: path to file with pretrained word embeddings
    :return: word2id, embedding matrix
    """
    with open(filename, encoding="UTF-8") as in_file:
        parts = in_file.readline().strip().split(" ")
        word_embedding_matrix = np.zeros(
            (int(parts[0]) + len(SPECIAL_TOKENS), int(parts[1]))
        )
        word2id = dict()
        for idx, token in enumerate(SPECIAL_TOKENS):
            word2id[token] = idx
        nextword_id = len(SPECIAL_TOKENS)
        for line in in_file:
            parts = line.strip().split(" ")
            word = parts[0]
            if word not in word2id:
                emb = [float(v) for v in parts[1:]]
                word_embedding_matrix[nextword_id] = emb
                word2id[word] = nextword_id
                nextword_id += 1
    return word2id, word_embedding_matrix


# deprecated
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
    parser.add_argument("--path_to_data", help="Path to the folder where all input files are stored.")
    parser.add_argument("--path_to_label_ids", help="Path to the file information about labels and their ids is stored")
    parser.add_argument("--path_to_word_embeddings", help="Path to the file with pretrained Glove embeddings used for "
                                                          "samples encoding")
    parser.add_argument("--sample_weights", help="Path to the folder that either sample weights will be saved to"
                                                 "or will be loaded from")
    parser.add_argument("--num_classes", help="Number of classes")

    args = parser.parse_args()

    train_wscrossweigh(
        args.path_to_data, args.path_labels, args.path_emb, args.sample_weights
    )
