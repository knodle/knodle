import argparse
import os
import sys

import torch
from torch import Tensor, LongTensor
from torch.optim import Adam
from torch.utils.data import TensorDataset
from transformers import RobertaForSequenceClassification, AdamW

from examples.trainer.ulf.ulf_training_bert_cosine import load_data, load_features, load_model
from knodle.trainer import WSCrossWeighTrainer
from knodle.trainer.wscw_wscl_ulf.wscrossweigh.config import WSCrossWeighConfig


def train_wscrossweigh_bert(path_to_data, dataset, model, output_path):
    path_to_data = os.path.join(path_to_data, dataset)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data from files
    df_train, df_dev, df_test, train_z, dev_z, test_z, t = load_data(path_to_data)
    model_name = load_model(model)

    # convert input data to features
    train_features, dev_features, test_features = load_features(df_train, df_dev, df_test, model, path_to_data, model_name, device)

    train_dataset = TensorDataset(Tensor(train_features))
    dev_dataset = TensorDataset(Tensor(dev_features))
    test_dataset = TensorDataset(Tensor(test_features))

    # gold labels for dev and test sets
    dev_labels = TensorDataset(LongTensor(df_dev["label"].tolist()))
    test_labels = TensorDataset(LongTensor(df_test["label"].tolist()))
    n_classes = max(max(df_test["label"].tolist()), max(df_dev["label"].tolist())) + 1

    parameters = {
        "lr": 1e-4, "cw_lr": 0.8, "epochs": 5, "cw_partitions": 2, "cw_folds": 5, "cw_epochs": 2, "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }
    # to have sample weights saved with some specific index in the file name, you can use "caching_suffix" variable
    caching_suffix = f"dscw_{parameters.get('cw_partitions')}part_{parameters.get('cw_folds')}folds_" \
                     f"{parameters.get('weight_rr')}wrr"

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=n_classes)

    custom_wscrossweigh_config = WSCrossWeighConfig(
        # general trainer parameters
        output_classes=n_classes,
        unmatched_strategy="filter",
        seed=12345,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=Adam,
        lr=parameters.get("lr"),
        grad_clipping=5,
        caching_suffix=caching_suffix,
        save_model_path=os.path.join(path_to_data, "trained_models"),
        # trained classifier model will be saved after each epoch

        # WSCrossWeigh specific parameters
        partitions=parameters.get("cw_partitions"),  # number of WSCrossWeigh iterations (= splitting into folds)
        folds=parameters.get("cw_folds"),  # number of folds train data will be splitted into
        weight_reducing_rate=parameters.get("weight_rr"),  # sample weights reducing coefficient
        samples_start_weights=parameters.get("samples_start_weights"),  # the start weight of sample weights
        cw_epochs=parameters.get("cw_epochs"),  # number of epochs each WSCrossWeigh model is to be trained
        cw_optimizer=Adam,  # WSCrossWeigh model optimiser
        cw_lr=parameters.get("cw_lr")  # WSCrossWeigh model lr
    )

    trainer = WSCrossWeighTrainer(
        # general Trainer inputs (a more detailed explanation of Knodle inputs is in README)
        model=model,  # classification model
        mapping_rules_labels_t=t,  # t matrix
        model_input_x=train_dataset,  # x matrix for training the classifier
        rule_matches_z=train_z,  # z matrix
        trainer_config=custom_wscrossweigh_config,

        # additional dev set used for classification model evaluation during training
        dev_model_input_x=dev_dataset,
        dev_gold_labels_y=dev_labels
    )

    # the WSCrossWeighTrainer is trained
    trainer.train()
    # the trained model is tested on the test set
    clf_report, _ = trainer.test(test_dataset, test_labels)
    print(clf_report)

    print("ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--dataset", help="")
    parser.add_argument("--model", help="")
    parser.add_argument("--output_path", help="")
    args = parser.parse_args()

    train_wscrossweigh_bert(args.path_to_data, args.dataset, args.model, args.output_path)