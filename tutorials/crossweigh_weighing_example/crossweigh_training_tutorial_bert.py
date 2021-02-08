import argparse
import os
import sys

import pandas as pd
import torch
from joblib import load
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, AdamW, AutoModelForSequenceClassification, BertForSequenceClassification, \
    DistilBertForSequenceClassification

from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.bert.crossweigh_with_bert import CrossWeigh

from tutorials.crossweigh_weighing_example.utils import vocab_and_vectors

NUM_CLASSES = 42
MAXLEN = 50
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])

# number of columns in input DataFrames with corresponding data
TRAIN_SAMPLES = 1       # in train DataFrame samples are stored in column with index 1
DEV_SAMPLES = 1
TEST_SAMPLES = 1
DEV_LABELS = 4
TEST_LABELS = 4


def train_crossweigh(
        path_t: str,
        path_train_samples: str,
        path_z: str,
        path_sample_weights: str = None,
        path_dev_features_labels: str = None,
        path_test_features_labels: str = None,
        path_labels: str = None
) -> None:

    os.makedirs(path_sample_weights, exist_ok=True)
    path_sample_weights = os.path.join(path_sample_weights, "0.8_2_15_2_0.3_2.0_True")

    if path_labels:
        labels2ids = read_labels_from_file(path_labels, "no_relation")

    rule_matches_z = load(path_z)
    rule_assignments_t = load(path_t)

    train_data = pd.read_csv(path_train_samples)
    dev_data = pd.read_csv(path_dev_features_labels)
    test_data = pd.read_csv(path_test_features_labels)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_input_x = get_features(train_data.iloc[:, TRAIN_SAMPLES], tokenizer, MAXLEN)

    if path_dev_features_labels:
        dev_dataset = get_features(dev_data.iloc[:, DEV_SAMPLES], tokenizer, MAXLEN)
        dev_labels = torch.LongTensor(list(dev_data.iloc[:, DEV_LABELS]))
    test_dataset = get_features(test_data.iloc[:, TEST_SAMPLES], tokenizer, MAXLEN)
    test_labels = torch.LongTensor(list(test_data.iloc[:, TEST_LABELS]))

    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_CLASSES)

    custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
        model=model,
        class_weights=CLASS_WEIGHTS,
        optimizer_=AdamW(model.parameters(), lr=0.01),
        output_classes=NUM_CLASSES
    )

    custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
        model=model,
        class_weights=CLASS_WEIGHTS,
        lr=1.0,
        output_classes=NUM_CLASSES,
        optimizer_=AdamW(model.parameters(), lr=0.01)
    )

    trainer = CrossWeigh(
        model=model,
        rule_assignments_t=rule_assignments_t,
        inputs_x=train_input_x,
        rule_matches_z=rule_matches_z,
        dev_features=test_dataset,
        dev_labels=test_labels,
        # evaluation_method="tacred",
        # dev_labels_ids=labels2ids,
        path_to_weights=path_sample_weights,
        denoising_config=custom_crossweigh_denoising_config,
        trainer_config=custom_crossweigh_trainer_config,
        use_weights=False
    )

    trainer.train()
    print("Testing on the test dataset....")

    # todo: add testing. add tacred eval. move def test to utils.py?


def get_features(data: pd.Series, tokenizer: DistilBertTokenizer, maxlen: int) -> TensorDataset:
    encoding = tokenizer(data.tolist(), return_tensors='pt', padding='max_length', max_length=maxlen, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    features = TensorDataset(input_ids, attention_mask)
    return features


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--rule_assignments_t", help="")
    parser.add_argument("--path_train_samples", help="")
    parser.add_argument("--rule_matches_z", help="")
    parser.add_argument("--sample_weights", help="")
    parser.add_argument("--dev_features_labels", help="")
    parser.add_argument("--test_features_labels", help="")
    parser.add_argument("--path_labels", help="")

    args = parser.parse_args()

    train_crossweigh(args.rule_assignments_t,
                     args.path_train_samples,
                     args.rule_matches_z,
                     args.sample_weights,
                     args.dev_features_labels,
                     args.test_features_labels,
                     args.path_labels)
