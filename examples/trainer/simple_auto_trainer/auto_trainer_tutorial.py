import os
from typing import List

from tqdm.auto import tqdm

import joblib
from minio import Minio

import pandas as pd
import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from examples.trainer.preprocessing import convert_text_to_transformer_input
from knodle.trainer import AutoTrainer, AutoConfig


# This python script contains rarely any explanation. For more description, we refer to the corresponding
# jupyter notebook. There the steps are explained in more detail


# Define some functions


def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    """

    :rtype: object
    """
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


# Define constants
imdb_data_dir = os.path.join(os.getcwd(), "datasets", "spouse")
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
        object_name=os.path.join("datasets/spouse/processed", file),
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

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

X_train = convert_text_to_transformer_input(tokenizer, df_train["sample"].tolist())
X_dev = convert_text_to_transformer_input(tokenizer, df_dev["sample"].tolist())
X_test = convert_text_to_transformer_input(tokenizer, df_test["sample"].tolist())

y_dev = np_array_to_tensor_dataset(df_dev['label'].values)
y_test = np_array_to_tensor_dataset(df_test['label'].values)

# Load AutoTrainer

model = AutoModelForSequenceClassification.from_pretrained(model_name)

trainer_type = "majority"
custom_model_config = AutoConfig.create_config(
    name=trainer_type,
    optimizer=AdamW,
    lr=1e-4,
    batch_size=16,
    epochs=2,
    filter_non_labelled=True
)

print(custom_model_config)

trainer = AutoTrainer(
    name="majority",
    model=model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=X_train,
    rule_matches_z=train_rule_matches_z,
    dev_model_input_x=X_dev,
    dev_gold_labels_y=y_dev,
    trainer_config=custom_model_config,
)

# Run training
trainer.train()

# Run evaluation
eval_dict, _ = trainer.test(X_test, y_test)
print(f"Accuracy: {eval_dict.get('accuracy')}")
