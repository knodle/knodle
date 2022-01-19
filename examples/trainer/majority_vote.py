#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import joblib
from torch import Tensor
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset

from examples.trainer.preprocessing import get_tfidf_features
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer import MajorityConfig, MajorityVoteTrainer

processed_data_dir = "../../data_from_minio/police_killing/processed_regex"
os.makedirs(processed_data_dir, exist_ok=True)
os.path.join(processed_data_dir)

# getting data
df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"))
df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))
mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))
train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
dev_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "dev_rule_matches_z.lib"))
test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))

# defining max features
X_train_tfidf, X_test_tfidf, X_dev_tfidf = get_tfidf_features(
    train_data=df_train["samples"].tolist(), dev_data=df_dev["samples"].tolist(),
    test_data=df_test["samples"].tolist(), max_features=5000
)

# converting features to datasets: gives a memory error when not setting max features to < 10000
X_train_tfidf_dataset = TensorDataset(Tensor(X_train_tfidf.toarray()))
X_dev_tfidf_dataset = TensorDataset(Tensor(X_dev_tfidf.toarray()))
X_test_tfidf_dataset = TensorDataset(Tensor(X_test_tfidf.toarray()))


# getting the labels
def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


y_dev = np_array_to_tensor_dataset(df_dev['enc_labels'].values)
y_test = np_array_to_tensor_dataset(df_test['enc_labels'].values)

NUM_OUTPUT_CLASSES = 2

model = LogisticRegressionModel(X_train_tfidf.shape[1], NUM_OUTPUT_CLASSES)
trainer_type = "majority"

# configuration
custom_model_config = MajorityConfig(  # AutoConfig.create_config(
    # name=trainer_type,
    lr=1e-4,
    batch_size=32,
    epochs=1,
    filter_non_labelled=False,
    other_class_id=0,
    use_probabilistic_labels=False
    # choose_random_label = False
)

# training
trainer = MajorityVoteTrainer(
    model=model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=X_train_tfidf_dataset,
    rule_matches_z=train_rule_matches_z,
    dev_model_input_x=X_dev_tfidf_dataset,
    dev_gold_labels_y=y_dev,
    trainer_config=custom_model_config
)

trainer.train()

trainer.test(X_test_tfidf_dataset, y_test)
