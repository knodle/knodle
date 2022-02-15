#!/usr/bin/env python
# coding: utf-8

# Imports
import os
import sys
from minio import Minio
import pandas as pd
import joblib
from torch import Tensor
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss

sys.path.append('../../..') # needed for Knodle imports

from knodle.trainer import KNNConfig, SnorkelConfig
from knodle.trainer.multi_trainer import MultiTrainer
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.config import MajorityConfig
from examples.trainer.preprocessing import get_tfidf_features


# set data directory
processed_data_dir = "../../../data_from_minio/police_killing/processed_kb"
os.makedirs(processed_data_dir, exist_ok=True)
os.path.join(processed_data_dir)

"""
# download pre-processed data from MinIO
client = Minio("knodle.cc", secure=False)
files = [
    "df_train.csv", "df_dev.csv", "df_test.csv",
    "mapping_rules_labels_t.lib", 
    "train_rule_matches_z.lib", "dev_rule_matches_z.lib", "test_rule_matches_z.lib"
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/police_killing/processed_kb", file),
        file_path=os.path.join(data_path, file),
    )

"""


# get the data
df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"))
df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))
mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))
train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
dev_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "dev_rule_matches_z.lib"))
test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))


# get tfidf features
X_train_tfidf, X_test_tfidf, X_dev_tfidf = get_tfidf_features(
    train_data=df_train["sample"].tolist(), test_data=df_test["sample"].tolist(),
    dev_data=df_dev["sample"].tolist(), max_features=5000
)


# convert features to datasets
X_train_tfidf_dataset = TensorDataset(Tensor(X_train_tfidf.toarray()))
X_dev_tfidf_dataset = TensorDataset(Tensor(X_dev_tfidf.toarray()))
X_test_tfidf_dataset = TensorDataset(Tensor(X_test_tfidf.toarray()))


# get the labels
def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x

y_dev = np_array_to_tensor_dataset(df_dev['enc_label'].values)
y_test = np_array_to_tensor_dataset(df_test['enc_label'].values)


# set number of output classes (positive and negative)
NUM_OUTPUT_CLASSES = 2

# set CLASS_WEIGHTS parameter due to unbalanced dataset
CLASS_WEIGHTS = torch.FloatTensor([0.5, 4.675])

# define the model
model = LogisticRegressionModel(X_train_tfidf.shape[1], NUM_OUTPUT_CLASSES)


# Trainer Configurations
configs = [
    MajorityConfig(
        lr=5e-1,
        batch_size=128,
        epochs=30,
        filter_non_labelled=False,
        other_class_id=0,
        use_probabilistic_labels=False,
        choose_random_label=False,
        criterion=CrossEntropyLoss,
        output_classes=2,
        class_weights = CLASS_WEIGHTS
    ),
    KNNConfig(
        lr=5e-1,
        batch_size=128,
        epochs=30,
        filter_non_labelled=False,
        other_class_id=0,
        use_probabilistic_labels=False,
        choose_random_label=False,
        criterion=CrossEntropyLoss,
        output_classes=2,
        class_weights = CLASS_WEIGHTS
    ),
    SnorkelConfig(
        lr=5e-1,
        batch_size=30,
        epochs = 30,
        filter_non_labelled=False,
        other_class_id=0,
        choose_random_label=False,
        output_classes=2,
        class_weights = CLASS_WEIGHTS,
        label_model_num_epochs = 30,
    ),
]

trainer = MultiTrainer(
    name=["majority", "knn", "snorkel"],
    model=model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=X_train_tfidf_dataset,
    rule_matches_z=train_rule_matches_z,
    dev_model_input_x=X_dev_tfidf_dataset,
    dev_gold_labels_y=y_dev,
    trainer_config=configs
)


# Run training
trainer.train()


# Run evaluation
metrics = trainer.test(X_test_tfidf_dataset, y_test)
print(metrics)