import argparse
import os
import sys
from itertools import product
from typing import Dict

import numpy as np
import pandas as pd
import torch
from joblib import load
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizer, AdamW, DistilBertForSequenceClassification

from knodle.evaluation.tacred_metrics import score
from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.trainer.crossweigh_weighing.crossweigh_bert import CrossWeigh
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from tutorials.crossweigh_weighing_example.utils import vocab_and_vectors


NUM_CLASSES = 42
NO_MATCH_CLASS_LABEL = 41
MAXLEN = 50
CLASS_WEIGHTS = torch.FloatTensor([2.0] * (NUM_CLASSES - 1) + [1.0])


def train_crossweigh(
        path_to_data: str,
        path_sample_weights: str = None,
        path_to_emb: str = None,
        path_to_labels: str = None,
) -> None:
    """
    Training the model with CrossWeigh model denoising
    :param path_to_data: path to a folder where all data is stored
    :param path_sample_weights: path to a folder where samples weights should be saved
    :param path_to_emb: path to file with pretrained embeddings
    :param path_to_labels: path to a file with class labels
    """

    labels2ids = read_labels_from_file(path_to_labels, "no_relation")
    word2id, word_embedding_matrix = vocab_and_vectors(path_to_emb)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_df, dev_df, test_df, rule_matches_z, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    train_input_x_glove = get_glove_encoded_features(train_df, word2id, 0)
    train_input_x_bert = get_bert_encoded_features(train_df, tokenizer, 0)

    dev_dataset_bert = get_bert_encoded_features(dev_df, tokenizer, 0)
    dev_labels = torch.LongTensor(list(dev_df.iloc[:, 3]))

    test_dataset_bert = get_bert_encoded_features(dev_df, tokenizer, 0)
    test_labels = torch.LongTensor(list(test_df.iloc[:, 3]))

    parameters = dict(
        cw_lr=[0.8],
        cw_partitions=[2, 3],
        cw_folds=[5, 10],
        cw_epochs=[2],
        weight_reducing_rate=[0.3],       # 0.7
        samples_start_weights=[2.0],        # 4.0
        epochs=[2]
    )
    param_values = [v for v in parameters.values()]

    tb = SummaryWriter('')

    for run_id, (cw_lr, cw_part, cw_folds, cw_epochs, weight_rr, start_weights, epochs) in \
            enumerate(product(*param_values)):
        comment = f'cw_lr = {cw_lr} cw_partitions = {cw_part} cw_folds = {cw_folds} cw_epochs = {cw_epochs} ' \
                  f'weight_reducing_rate = {weight_rr} samples_start_weights = {start_weights} epochs = {epochs}'

        tb = SummaryWriter(comment=comment)
        folder_prefix = f'{cw_lr}_{cw_part}_{cw_folds}_{cw_epochs}_{weight_rr}_{start_weights}'
        path_to_weights = os.path.join(path_sample_weights, folder_prefix)
        os.makedirs(path_to_weights, exist_ok=True)

        print("Parameters: {}".format(comment))
        print("Weights will be saved to/loaded from {}".format(folder_prefix))

        model_cw = BidirectionalLSTM(
            word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], word_embedding_matrix, NUM_CLASSES
        )

        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=NUM_CLASSES
        )

        custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
            model=model_cw,
            crossweigh_partitions=cw_part,
            class_weights=CLASS_WEIGHTS,
            crossweigh_folds=cw_folds,
            crossweigh_epochs=cw_epochs,
            weight_reducing_rate=weight_rr,
            samples_start_weights=start_weights,
            optimizer_=torch.optim.Adam(model.parameters(), lr=cw_lr),
            output_classes=NUM_CLASSES,
            batch_size=512,
            filter_empty_probs=False,
            no_match_class_label=NO_MATCH_CLASS_LABEL
        )
        custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
            model=model,
            class_weights=CLASS_WEIGHTS,
            output_classes=NUM_CLASSES,
            optimizer_=AdamW(model.parameters(), lr=1e-4),
            epochs=epochs,
            batch_size=32,
            filter_empty_probs=False,
            no_match_class_label=NO_MATCH_CLASS_LABEL
        )

        trainer = CrossWeigh(
            model=model,
            rule_assignments_t=mapping_rules_labels_t,
            inputs_x=train_input_x_bert,
            rule_matches_z=rule_matches_z,
            dev_features=dev_dataset_bert,
            dev_labels=dev_labels,
            cw_model=model_cw,
            cw_inputs_x=train_input_x_glove,
            evaluation_method="tacred",
            dev_labels_ids=labels2ids,
            path_to_weights=path_to_weights,
            denoising_config=custom_crossweigh_denoising_config,
            trainer_config=custom_crossweigh_trainer_config,
            run_classifier=True
        )
        trainer.train()
        print("Testing on the test dataset....")
        metrics = test_tacred_dataset(model, trainer, test_dataset_bert, test_labels, labels2ids)
        # metrics = test(model, optimizer, features_dataset: TensorDataset, labels: TensorDataset, device)

        # tb.add_hparams(
        #     {"cw_lr": cw_lr,
        #      "epochs": epochs,
        #      "cw_partitions": cw_part,
        #      "cw_folds": cw_folds,
        #      "cw_epochs": cw_epochs,
        #      "weight_reducing_rate": weight_rr,
        #      "samples_start_weights": start_weights},
        #     {"precision": metrics["precision"],
        #      "recall": metrics["recall"],
        #      "f1": metrics["f1"]})

        tb.close()
        print("========================================================================")
        print("========================================================================")
        print("========================== RUN {} IS DONE ==============================".format(run_id))
        print("========================================================================")


def read_train_dev_test(target_path: str):
    train_df = load(os.path.join(target_path, 'df_train.lib'))
    dev_df = load(os.path.join(target_path, 'df_dev.lib'))
    test_df = load(os.path.join(target_path, 'df_test.lib'))
    train_rule_matches_z = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(target_path, 'mapping_rules_labels.lib'))

    return train_df, dev_df, test_df, train_rule_matches_z, mapping_rules_labels_t


def get_glove_encoded_features(input_data: pd.DataFrame, word2id: dict, column_num: int) -> TensorDataset:
    enc_input_samples = encode_samples(list(input_data.iloc[:, column_num]), word2id, MAXLEN)
    return torch.utils.data.TensorDataset(torch.LongTensor(enc_input_samples))


def get_bert_encoded_features(input_data: pd.Series, tokenizer: DistilBertTokenizer, column_num: int) -> TensorDataset:
    encoding = tokenizer(list(input_data.iloc[:, column_num]), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return TensorDataset(input_ids, attention_mask)


def read_labels_from_file(path_labels: str, negative_label: str) -> dict:
    """ Reads the labels from the file and encode them with ids """
    relation2ids = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            relation2ids[relation] = int(relation_enc)

    # add no_match label
    if negative_label:
        relation2ids[negative_label] = NO_MATCH_CLASS_LABEL

    return relation2ids


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


def test_tacred_dataset(model, trainer, test_features: TensorDataset, test_labels: Tensor, labels2ids: Dict) -> Dict:
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
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--sample_weights", help="")
    parser.add_argument("--path_to_emb", help="")
    parser.add_argument("--path_to_labels", help="")

    args = parser.parse_args()

    train_crossweigh(
        args.path_to_data,
        args.sample_weights,
        args.path_to_emb,
        args.path_to_labels
    )
