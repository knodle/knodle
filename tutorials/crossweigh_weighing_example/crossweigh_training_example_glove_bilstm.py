import argparse
import os
import sys
from itertools import product
from typing import Dict

import pandas as pd
import numpy as np
import scipy
import torch
from joblib import load
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchtext.vocab import GloVe
from tqdm import tqdm

from knodle.evaluation.tacred_metrics import score
from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeigh
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from tutorials.crossweigh_weighing_example.utils import encode_samples, vocab_and_vectors

NUM_CLASSES = 42
MAXLEN = 50
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
NO_MATCH_CLASS_LABEL = 41


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
    :param path_t: path to binary matrix that contains info about which rule corresponds to which label (rule x labels)
    :param path_dev_features_labels: path to DataFrame with development data (1st column - samples, 2nd column - labels)
    """

    labels2ids = read_labels_from_file(path_labels, "no_relation")

    word2id, word_embedding_matrix = vocab_and_vectors(path_emb)

    rule_matches_z = load(path_z)
    rule_matches_z = rule_matches_z.toarray() if scipy.sparse.issparse(rule_matches_z) else rule_matches_z
    rule_assignments_t = load(path_t)
    train_input_x = get_train_features(path_train_samples, word2id, 1)

    dev_dataset, dev_labels = get_dev_data(path_dev_features_labels, word2id)
    test_dataset, test_labels = get_dev_data(path_test_features_labels, word2id)

    parameters = dict(
        cw_lr=[0.8],       # 0.01,
        cw_epochs=[2],
        weight_reducing_rate=[0.3],      #  0.3, 0.7
        samples_start_weights=[0],       # 2.0, 3.0
        cw_partitions=[2],         # 2, 3
        cw_folds=[10],       # 5, 10
        use_weights=[True],         # True, False
        epochs=[3, 5],        # 25, 35, 50, 100
        lr=[2.0, 0.8]
    )
    param_values = [v for v in parameters.values()]

    tb = SummaryWriter('')

    for run_id, (cw_lr, cw_epochs, weight_rr, start_weights, cw_part, cw_folds, use_weights, epochs, lr) in \
            enumerate(product(*param_values)):
        comment = f'lr = {lr} cw_lr = {cw_lr} epochs = {epochs} cw_partitions = {cw_part} cw_folds = {cw_folds} ' \
                  f'cw_epochs = {cw_epochs} weight_reducing_rate = {weight_rr} samples_start_weights {start_weights} ' \
                  f'use_weights = {use_weights}'

        # print(comment)
        if comment == "lr = 0.1 cw_lr = 0.8 epochs = 3 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.7 samples_start_weights 2.0 use_weights = True":
            # print("This experiment was already done; skip it")
            continue
        if comment == "lr = 0.1 cw_lr = 0.8 epochs = 5 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.7 samples_start_weights 2.0 use_weights = True":
            # print("This experiment was already done; skip it")
            continue
        if comment == "lr = 0.8 cw_lr = 0.8 epochs = 3 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.7 samples_start_weights 2.0 use_weights = True":
            continue

        if comment == "lr = 0.8 cw_lr = 0.8 epochs = 3 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.3 samples_start_weights 2.0 use_weights = True":
            continue

        if comment == "lr = 2.0 cw_lr = 0.8 epochs = 3 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.3 samples_start_weights 2.0 use_weights = True":
            continue

        if comment == "lr = 0.8 cw_lr = 0.8 epochs = 5 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.3 samples_start_weights 2.0 use_weights = True":
            continue

        if comment == "lr = 2.0 cw_lr = 0.8 epochs = 5 cw_partitions = 2 cw_folds = 5 cw_epochs = 2 " \
                      "weight_reducing_rate = 0.3 samples_start_weights 2.0 use_weights = True":
            continue

        tb = SummaryWriter(comment=comment)
        folder_prefix = f'{cw_lr}_{cw_part}_{cw_folds}_{cw_epochs}_{weight_rr}_{start_weights}_{use_weights}'
        path_to_weights = os.path.join(path_sample_weights, folder_prefix)
        os.makedirs(path_to_weights, exist_ok=True)

        with open(path_to_weights + "_" + str(epochs) + "_" + str(lr) + "_" + str(run_id) + "_results.txt", 'w+') as f:
            sys.stdout = f

            print("Parameters: {}".format(comment))
            print("Weights will be saved to/loaded from {}".format(folder_prefix))

            model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                                      word_embedding_matrix.shape[1],
                                      word_embedding_matrix,
                                      NUM_CLASSES)

            custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(model=model,
                                                                           crossweigh_partitions=cw_part,
                                                                           class_weights=CLASS_WEIGHTS,
                                                                           crossweigh_folds=cw_folds,
                                                                           crossweigh_epochs=cw_epochs,
                                                                           weight_reducing_rate=weight_rr,
                                                                           samples_start_weights=start_weights,
                                                                           lr=cw_lr,
                                                                           optimizer_=torch.optim.Adam(model.parameters()),
                                                                           output_classes=NUM_CLASSES
                                                                           )
            custom_crossweigh_trainer_config = CrossWeighTrainerConfig(model=model,
                                                                       class_weights=CLASS_WEIGHTS,
                                                                       lr=lr,
                                                                       output_classes=NUM_CLASSES,
                                                                       optimizer_=torch.optim.Adam(model.parameters()),
                                                                       epochs=epochs
                                                                       )

            trainer = CrossWeigh(model=model,
                                 rule_assignments_t=rule_assignments_t,
                                 inputs_x=train_input_x,
                                 rule_matches_z=rule_matches_z,
                                 dev_features=dev_dataset,
                                 dev_labels=dev_labels,
                                 evaluation_method="tacred",
                                 dev_labels_ids=labels2ids,
                                 path_to_weights=path_to_weights,
                                 denoising_config=custom_crossweigh_denoising_config,
                                 trainer_config=custom_crossweigh_trainer_config,
                                 run_classifier=True,
                                 use_weights=use_weights
                                 )
            trainer.train()
            print("Testing on the test dataset....")
            # metrics = test(model, trainer, test_dataset, test_labels, labels2ids)
            metrics = test(model, optimizer, features_dataset: TensorDataset, labels: TensorDataset, device)

            tb.add_hparams(
                {"lr": lr,
                 "cw_lr": cw_lr,
                 "epochs": epochs,
                 "cw_partitions": cw_part,
                 "cw_folds": cw_folds,
                 "cw_epochs": cw_epochs,
                 "weight_reducing_rate": weight_rr,
                 "samples_start_weights": start_weights,
                 "use_sample_weights": use_weights},
                {"precision": metrics["precision"],
                 "recall": metrics["recall"],
                 "f1": metrics["f1"]})

            tb.close()
            print("========================================================================")
            print("========================================================================")
            print("========================== RUN {} IS DONE ==============================".format(run_id))
            print("========================================================================")

            sys.stdout.close()


def get_train_features(path_train_data: str, word2id: dict, column_num: int) -> TensorDataset:
    """
    This function reads the input data saved as a DataFrame and encode sentences with words ids.
    :param path_train_data: path to .csv file with input data
    :param word2id: dictionary of words to their ids that corresponds to pretrained embeddings
    :param column_num: number of a column in DataFrame with input data where samples are stored
    :return:
    """
    input_data = pd.read_csv(path_train_data)
    enc_input_samples = encode_samples(list(input_data.iloc[:, column_num]), word2id, MAXLEN)
    inputs_x_tensor = torch.LongTensor(enc_input_samples)
    inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)
    return inputs_x_dataset


def get_dev_data(path_dev_feature_labels: str, word2id: dict) -> TensorDataset:
    """ Read dev data with gold labels and turn it into TensorDataset(features, labels)"""
    dev_data = pd.read_csv(path_dev_feature_labels)
    enc_dev_samples = encode_samples(list(dev_data.iloc[:, 1]), word2id, MAXLEN)
    dev_samples_dataset = torch.utils.data.TensorDataset(torch.LongTensor(enc_dev_samples))
    dev_labels_tensor = torch.LongTensor(list(dev_data.iloc[:, 4]))
    return dev_samples_dataset, dev_labels_tensor


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


# def create_tfidf_values(train_data: [str], test_data: [str]):
#     if os.path.exists("tfidf.lib"):
#         cached_data = load("tfidf.lib")
#         if cached_data.shape == train_data.shape:
#             print("Cashed tfidf values are loaded")
#             return cached_data
#
#     print("New tfidf values will be calculated")
#     vectorizer = TfidfVectorizer()
#     train_transformed_data = vectorizer.fit_transform(train_data)
#     dump(train_transformed_data, "tfidf.lib")
#
#     test_transformed_data = vectorizer.transform(test_data)
#     return train_transformed_data, test_transformed_data


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