import json
import logging
import os

import pandas as pd
import torch
from joblib import load

from examples.trainer.ulf.utils import bert_text_extractor, z_to_wrench_labels, WrenchDataset
from knodle.trainer.utils.utils import set_seed
from wrench.endmodel import Cosine

logger = logging.getLogger(__name__)

lr = 0.001
n_steps = 100000
batch_size = 128
test_batch_size = 32
patience = 200
evaluation_step = 50
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 12345
set_seed(seed)

dataset = 'youtube'
target = 'acc'
path = '../../../data_from_minio/wrench_knodle_format'
path_to_data = os.path.join(path, dataset)
label_path = os.path.join('../../../data_from_minio/wrench', dataset, 'label.json')

# Load data
df_train = pd.read_csv(os.path.join(path_to_data, 'train_df.csv'), sep="\t")
df_dev = pd.read_csv(os.path.join(path_to_data, 'dev_df.csv'), sep="\t")
df_test = pd.read_csv(os.path.join(path_to_data, 'test_df.csv'), sep="\t")
rule_matches_z_train = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
rule_matches_z_dev = load(os.path.join(path_to_data, 'dev_rule_matches_z.lib'))
rule_matches_z_test = load(os.path.join(path_to_data, 'test_rule_matches_z.lib'))
mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))

extract_fn = 'bert'
model_name = 'roberta-base'

# get bert features
train_features = bert_text_extractor(data=list(df_train["sample"]), path=path_to_data, split="train", cache_name=extract_fn, model_name=model_name)
dev_features = bert_text_extractor(data=list(df_dev["sample"]), path=path_to_data, split="dev", cache_name=extract_fn, model_name=model_name)
test_features = bert_text_extractor(data=list(df_test["sample"]), path=path_to_data, split="test", cache_name=extract_fn, model_name=model_name)

# get weak labels in wrench/snorkel format
train_weak_labels = z_to_wrench_labels(rule_matches_z_train, mapping_rules_labels_t)
dev_weak_labels = z_to_wrench_labels(rule_matches_z_dev, mapping_rules_labels_t)
test_weak_labels = z_to_wrench_labels(rule_matches_z_test, mapping_rules_labels_t)

# gold labels for dev and test sets
dev_labels = df_dev["label"].tolist()
test_labels = df_test["label"].tolist()

n_classes = max(test_labels) + 1
n_lf = mapping_rules_labels_t.shape[0]
id2label = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}

# calculate noisy labels
# noisy_y_train = z_t_matrices_to_majority_vote_probs(rule_matches_z_train, mapping_rules_labels_t)
# noisy_y_train = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=noisy_y_train)

# transform data to wrench datasets
train_data = WrenchDataset(features=train_features, n_class=n_classes, n_lf=n_lf,
                           examples=list(df_train["sample"]), weak_labels=train_weak_labels, id2label=id2label)
dev_data = WrenchDataset(features=dev_features, n_class=n_classes, n_lf=n_lf, labels=dev_labels,
                         examples=list(df_dev["sample"]), weak_labels=dev_weak_labels, id2label=id2label)
test_data = WrenchDataset(features=test_features, n_class=n_classes, n_lf=n_lf, labels=test_labels,
                          examples=list(df_test["sample"]), weak_labels=test_weak_labels, id2label=id2label)

model = Cosine(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size, optimizer_lr=lr,
               optimizer_weight_decay=1e-4, thresh=0.4, lamda=0.05, mu=1)
model.fit(dataset_train=train_data, dataset_valid=dev_data, device=device, metric=target, patience=patience,
          evaluation_step=evaluation_step, seed=seed)

# Evaluate the trained model
metric_value = model.test(test_data, target)

print(metric_value)
