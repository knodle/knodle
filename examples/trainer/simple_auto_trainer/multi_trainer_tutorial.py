import json
import os
import statistics
from sys import path

from scipy.stats import sem
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm.auto import tqdm

import joblib
import shutil
from minio import Minio

import pandas as pd
import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import TensorDataset
from transformers import AdamW

from examples.trainer.preprocessing import get_tfidf_features
from knodle.trainer import MajorityConfig, KNNConfig, SnorkelConfig, SnorkelKNNConfig
from knodle.model.logistic_regression_model import LogisticRegressionModel

from random import randint

# This python script contains rarely any explanation. For more description, we refer to the corresponding
# jupyter notebook. There the steps are explained in more detail


# Define some functions
from knodle.trainer.multi_trainer import MultiTrainer
from knodle.trainer.wscrossweigh.config import WSCrossWeighConfig


def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


# Define constants
# imdb_data_dir = os.path.join(os.getcwd(), "datasets", "spam")
# processed_data_dir = os.path.join(imdb_data_dir, "processed")
# os.makedirs(processed_data_dir, exist_ok=True)

processed_data_dir = "/Users/asedova/PycharmProjects/01_knodle/data_from_minio/trec/processed"

# Download data
client = Minio("knodle.cc", secure=False)
files = [
    "df_train.csv", "df_test.csv", "df_dev.csv",
    "train_rule_matches_z.lib", "test_rule_matches_z.lib",
    "mapping_rules_labels_t.lib"
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/trec/processed", file),
        file_path=os.path.join(processed_data_dir, file),
    )

# Load data into memory
df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))
df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"))

mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))

train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))

print(f"Train Z dimension: {train_rule_matches_z.shape}")
print(f"Train avg. matches per sample: {train_rule_matches_z.sum() / train_rule_matches_z.shape[0]}")

# tfidf
X_train_tfidf, X_test_tfidf, X_dev_tfidf = get_tfidf_features(
    train_data=df_train["sample"].tolist(),
    test_data=df_test["sample"].tolist(),
    dev_data=df_dev["sample"].tolist()
)
# convert input features to datasets
X_train_tfidf_dataset = TensorDataset(Tensor(X_train_tfidf.toarray()))
X_test_tfidf_dataset = TensorDataset(Tensor(X_test_tfidf.toarray()))
X_dev_tfidf_dataset = TensorDataset(Tensor(X_dev_tfidf.toarray()))

# get test labels
y_test = np_array_to_tensor_dataset(df_test['label'].values)
y_dev = np_array_to_tensor_dataset(df_dev['label'].values)

num_classes = max(df_test["label"].tolist()) + 1

num_experiments = 10
res_snorkel, res_crossweigh, res_majority = [], [], []

for i in range(num_experiments):

    seed = randint(1, 300)
    # initilize model
    logreg_model = LogisticRegressionModel(X_train_tfidf.shape[1], num_classes)

    configs = [
        # MajorityConfig(
        #     output_classes=num_classes, optimizer=Adam, use_probabilistic_labels=False, criterion=CrossEntropyLoss,
        #     lr=0.1, batch_size=256, epochs=15, seed=seed
        # ),
        SnorkelConfig(
            seed=seed, optimizer=Adam, lr=0.001, epochs=15, batch_size=256, output_classes=num_classes,
            use_probabilistic_labels=False, criterion=CrossEntropyLoss
            # filter_non_labelled=True
            # verbose=False
        ),
        WSCrossWeighConfig(
            optimizer=Adam, folds=2, partitions=25, weight_reducing_rate=0.3, output_classes=num_classes,
            criterion=CrossEntropyLoss, filter_non_labelled=False, lr=0.05, epochs=20, batch_size=128, grad_clipping=5,
            early_stopping=True, use_probabilistic_labels=False, verbose=False, seed=seed
        ),
        # SnorkelKNNConfig(optimizer=AdamW, radius=0.8),
        # KNNConfig(optimizer=AdamW, k=2, lr=1e-4, batch_size=32, epochs=2),
    ]
    # print([config.__dict__ for config in configs])

    trainer = MultiTrainer(
        # name=["majority", "knn", "snorkel", "snorkel_knn", "wscrossweigh"],
        name=["snorkel"],
        model=logreg_model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=X_train_tfidf_dataset,
        rule_matches_z=train_rule_matches_z,
        trainer_config=configs,
        dev_model_input_x=X_dev_tfidf_dataset,
        dev_gold_labels_y=y_dev
    )

    # Run training
    trainer.train()

    # Run evaluation
    metrics = trainer.test(X_test_tfidf_dataset, y_test)
    for trainer, metric in metrics.items():
        if trainer == "SnorkelTrainer":
            # res_snorkel.append(metric.get('macro avg')['f1-score'])
            # print(f"Trainer: {trainer}, {'F1'}: {metric.get('macro avg')['f1-score']}")
            res_snorkel.append(metric.get('accuracy'))
            print(f"Trainer: {trainer}, {'F1'}: {metric.get('accuracy')}")
        elif trainer == "WSCrossWeighTrainer":
            # res_crossweigh.append(metric.get('macro avg')['f1-score'])
            # print(f"Trainer: {trainer}, {'F1'}: {metric.get('macro avg')['f1-score']}")
            res_crossweigh.append(metric.get('accuracy'))
            print(f"Trainer: {trainer}, {'F1'}: {metric.get('accuracy')}")
        elif trainer == "MajorityVoteTrainer":
            # res_majority.append(metric.get('macro avg')['f1-score'])
            # print(f"Trainer: {trainer}, {'F1'}: {metric.get('macro avg')['f1-score']}")
            res_majority.append(metric.get('accuracy'))
            print(f"Trainer: {trainer}, {'F1'}: {metric.get('accuracy')}")
        else:
            raise ValueError("Unknown Trainer!")

    if os.path.isdir('/Users/asedova/PycharmProjects/01_knodle/examples/trainer/simple_auto_trainer/cache'):
        shutil.rmtree('/Users/asedova/PycharmProjects/01_knodle/examples/trainer/simple_auto_trainer/cache')

majority_mean = statistics.mean(res_majority)
majority_stdev = statistics.stdev(res_majority)
majority_sem = sem(res_majority)

snorkel_mean = statistics.mean(res_snorkel)
snorkel_stdev = statistics.stdev(res_snorkel)
snorkel_sem = sem(res_snorkel)

# crossweigh_mean = statistics.mean(res_crossweigh)
# crossweigh_stdev = statistics.stdev(res_crossweigh)
# crossweigh_sem = sem(res_crossweigh)

for i in range(len(configs)):
    del configs[i].__dict__["criterion"]
    del configs[i].__dict__["optimizer"]
    del configs[i].__dict__["device"]
    del configs[i].__dict__["class_weights"]

res_majority = {
    "values": res_majority,
    "mean": majority_mean,
    "stdev": majority_stdev,
    "sem": majority_sem
}
res_majority = {**res_majority, **configs[0].__dict__}
print(res_majority)

res_snorkel = {
    "values": res_snorkel,
    "mean": snorkel_mean,
    "stdev": snorkel_stdev,
    "sem": snorkel_sem
}
res_snorkel = {**res_snorkel, **configs[1].__dict__}
print(res_snorkel)
#
# res_crossweigh = {
#     "values": res_crossweigh,
#     "mean": crossweigh_mean,
#     "stdev": crossweigh_stdev,
#     "sem": crossweigh_sem
# }
# res_crossweigh = {**res_crossweigh, **configs[2].__dict__}
# print(res_crossweigh)


with open(os.path.join(processed_data_dir, "snorkel_res_3.json"), 'w') as file:
    json.dump(res_snorkel, file)

# with open(os.path.join(processed_data_dir, "crossweigh_res.json"), 'w') as file:
#     json.dump(res_crossweigh, file)

with open(os.path.join(processed_data_dir, "majority_res_3.json"), 'w') as file:
    json.dump(res_majority, file)

