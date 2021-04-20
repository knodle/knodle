import os
from minio import Minio
from tqdm.auto import tqdm

import joblib
import pandas as pd

from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.auto_trainer import AutoTrainer

# This python script contains rarely any explanation. For more description, we refer to the corresponding
# jupyter notebook. There the steps are explained in more detail


# Define some functions

def remove_stop_words(text: str) -> str:
    text = ' '.join([word for word in text.split() if word not in (ENGLISH_STOP_WORDS)])
    return text


def np_array_to_tensor_dataset(x: np.ndarray):
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


def create_tfidf_dataset(
        text_data: [str], force_create_new: bool = False, max_features: int = None
):
    """Takes a list of strings, e.g. sentences, and transforms these in a simple TF-IDF representation"""
    text_data = [remove_stop_words(t) for t in text_data]
    vectorizer = TfidfVectorizer(min_df=2, max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data)
    dataset = np_array_to_tensor_dataset(transformed_data)
    return dataset


# Define constants
imdb_data_dir = os.path.join(os.getcwd(), "data", "imdb")
processed_data_dir = os.path.join(imdb_data_dir, "processed")
os.makedirs(processed_data_dir, exist_ok=True)

# Download data
client = Minio("knodle.dm.univie.ac.at", secure=False)
files = [
    "df_train.csv", "df_dev.csv", "df_test.csv",
    "train_rule_matches_z.lib", "dev_rule_matches_z.lib", "test_rule_matches_z.lib",
    "mapping_rules_labels_t.lib"
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/imdb/processed", file),
        file_path=os.path.join(processed_data_dir, file),
    )

# Load data into memory
df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"))
df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))

mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))

train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
dev_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "dev_rule_matches_z.lib"))
test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))

print(f"Train Z dimension: {train_rule_matches_z.shape}")
print(f"Train avg. matches per sample: {train_rule_matches_z.sum() / train_rule_matches_z.shape[0]}")

# Preprocess data
X_train = create_tfidf_dataset(df_train["sample"].tolist(), max_features=5000)
X_dev = create_tfidf_dataset(df_dev["sample"].tolist(), max_features=5000)
X_test = create_tfidf_dataset(df_test["sample"].tolist(), max_features=5000)

y_dev = np_array_to_tensor_dataset(df_dev['label'].values)
y_test = np_array_to_tensor_dataset(df_test['label'].values)

# Load AutoTrainer
model = LogisticRegressionModel(X_train.tensors[0].shape[1], 2)

maj_trainer = AutoTrainer(
    name="majority",
    model=model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=X_train,
    rule_matches_z=train_rule_matches_z,
    dev_model_input_x=X_dev,
    dev_gold_labels_y=y_dev
)

# Start training
maj_trainer.train()

# Run evaluation
eval_dict, _ = maj_trainer.test(X_test, y_test)
print(f"Accuracy: {eval_dict.get('accuracy')}")

