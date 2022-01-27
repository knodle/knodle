#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import joblib
from torch import Tensor
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss

from knodle.trainer import KNNConfig, SnorkelConfig
from knodle.trainer.multi_trainer import MultiTrainer

sys.path.append('../..')

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.baseline.config import MajorityConfig
from examples.trainer.preprocessing import get_tfidf_features

processed_data_dir = "../../data_from_minio/police_killing/processed_kb"
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

# changing datatype
mapping_rules_labels_t = mapping_rules_labels_t.toarray()
mapping_rules_labels_t = mapping_rules_labels_t.astype('int64')
mapping_rules_labels_t = sp.csr_matrix(mapping_rules_labels_t)

train_rule_matches_z = train_rule_matches_z.toarray()
train_rule_matches_z = train_rule_matches_z.astype('int64')
train_rule_matches_z = sp.csr_matrix(train_rule_matches_z)

dev_rule_matches_z = dev_rule_matches_z.toarray()
dev_rule_matches_z = dev_rule_matches_z.astype('int64')
dev_rule_matches_z = sp.csr_matrix(dev_rule_matches_z)

# defining max features
X_train_tfidf, X_test_tfidf, X_dev_tfidf = get_tfidf_features(
    train_data=df_train["sample"].tolist(), test_data=df_test["sample"].tolist(),
    dev_data=df_dev["sample"].tolist(), max_features=1000
)

# converting features to datasets: gives a memory error when not setting max features to < 10000
# X_train_tfidf_dataset = TensorDataset(Tensor(X_train_tfidf.toarray()))
X_train_tfidf = Tensor(X_train_tfidf.toarray())
X_train_tfidf = torch.tensor(X_train_tfidf, dtype=torch.long)
X_train_tfidf_dataset = TensorDataset(X_train_tfidf)
# X_dev_tfidf_dataset = TensorDataset(Tensor(X_dev_tfidf.toarray()))
X_dev_tfidf = Tensor(X_dev_tfidf.toarray())
X_dev_tfidf = torch.tensor(X_dev_tfidf, dtype=torch.long)
X_dev_tfidf_dataset = TensorDataset(X_dev_tfidf)
# X_test_tfidf_dataset = TensorDataset(Tensor(X_test_tfidf.toarray()))
X_test_tfidf = Tensor(X_test_tfidf.toarray())
X_test_tfidf = torch.tensor(X_test_tfidf, dtype=torch.long)
X_test_tfidf_dataset = TensorDataset(X_test_tfidf)


# getting the labels
def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


y_dev = np_array_to_tensor_dataset(df_dev['enc_label'].values)
y_test = np_array_to_tensor_dataset(df_test['enc_label'].values)

# changing datatype
# y_dev = torch.tensor(y_dev, dtype=torch.long)


NUM_OUTPUT_CLASSES = 2

model = LogisticRegressionModel(X_train_tfidf.shape[1], NUM_OUTPUT_CLASSES)

configs = [
    MajorityConfig(
        lr=1e-3,
        batch_size=128,
        epochs=20,
        filter_non_labelled=False,
        other_class_id=0,
        use_probabilistic_labels=False,
        choose_random_label=False,
        criterion=CrossEntropyLoss,
        output_classes=2
    ),
    KNNConfig(
        lr=1e-3,
        batch_size=128,
        epochs=20,
        filter_non_labelled=False,
        other_class_id=0,
        use_probabilistic_labels=False,
        choose_random_label=False,
        criterion=CrossEntropyLoss,
        output_classes=2
    ),
    SnorkelConfig(
        lr=1e-3,
        batch_size=128,
        epochs=20,
        filter_non_labelled=False,
        other_class_id=0,
        choose_random_label=False,
        output_classes=2
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
for trainer, metric in metrics.items():
    print(f"Trainer: {trainer}, accuracy: {metric[0].get('accuracy')}")
