import os
from torch import Tensor
from tqdm.auto import tqdm

import joblib
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
imdb_data_dir = os.path.join(os.getcwd(), "datasets", "spam")
processed_data_dir = os.path.join(imdb_data_dir, "processed")
os.makedirs(processed_data_dir, exist_ok=True)

# Download data
client = Minio("knodle.dm.univie.ac.at", secure=False)
files = [
    "df_train.csv", "df_test.csv",
    "train_rule_matches_z.lib", "test_rule_matches_z.lib",
    "mapping_rules_labels_t.lib"
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/spam/processed", file),
        file_path=os.path.join(processed_data_dir, file),
    )

# Load data into memory
df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))

mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))

train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))

print(f"Train Z dimension: {train_rule_matches_z.shape}")
print(f"Train avg. matches per sample: {train_rule_matches_z.sum() / train_rule_matches_z.shape[0]}")

# tfidf
X_train_tfidf, X_test_tfidf, _ = get_tfidf_features(
    train_data=df_train["sample"].tolist(),
    test_data=df_test["sample"].tolist()
)
# convert input features to datasets
X_train_tfidf_dataset = TensorDataset(Tensor(X_train_tfidf.toarray()))
X_test_tfidf_dataset = TensorDataset(Tensor(X_test_tfidf.toarray()))

# get test labels
y_test = np_array_to_tensor_dataset(df_test['label'].values)

# initilize model
logreg_model = LogisticRegressionModel(X_train_tfidf.shape[1], 2)

configs = [
    MajorityConfig(optimizer=AdamW, lr=1e-4, batch_size=16, epochs=3),
    KNNConfig(optimizer=AdamW, k=2, lr=1e-4, batch_size=32, epochs=2),
    SnorkelConfig(optimizer=AdamW),
    SnorkelKNNConfig(optimizer=AdamW, radius=0.8),
    WSCrossWeighConfig(optimizer=AdamW)
]


trainer = MultiTrainer(
    name=["majority", "knn", "snorkel", "snorkel_knn", "wscrossweigh"],
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=X_train_tfidf_dataset,
    rule_matches_z=train_rule_matches_z,
    trainer_config=configs,
)

# Run training
trainer.train()

# Run evaluation
metrics = trainer.test(X_test_tfidf_dataset, y_test)
for trainer, metric in metrics.items():
    print(f"Trainer: {trainer}, accuracy: {metric[0].get('accuracy')}")
