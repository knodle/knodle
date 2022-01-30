from torch.nn import BCEWithLogitsLoss
import numpy as np
import os
import torch
from joblib import load
from transformers import AdamW
from torch.utils.data import TensorDataset
from knodle.model.logistic_regression_model import LogisticRegressionModel
# from knodle.trainer import MajorityVoteTrainer, MajorityConfig
#from knodle.trainer.baseline.majority import MajorityVoteTrainer
#from knodle.trainer.baseline.config import MajorityConfig
# from knodle.trainer import MajorityConfig, MajorityVoteTrainer
# from knodle.trainer import AutoTrainer, AutoConfig
# from knodle.trainer import MajorityVoteTrainer, MajorityConfig
os.getcwd()
os.chdir('C:\\Users\\marli\\physionet.org')
Z = "rule_matches_z.lib"
T = "mapping_rules_labels_t.lib"
X = "train_X.lib"
X_finetuned = "train_X_finetuned.lib"
input_l = "input_list.lib"
X_test = "X_test.lib"
X_test_finetuned = "X_test_finetuned.lib"
y_test = "y_gold_test.lib"


rule_matches_z = load(Z)
mapping_rules_labels_t = load(T)
train_X = load(X)
train_X_finetuned = load(X_finetuned)
test_X = load(X_test)
gold_labels_test = load(y_test)
test_X_finetuned = load(X_test_finetuned)
gold_labels_test = load(y_test)

# how many without matches
sum(np.sum(rule_matches_z, axis = 1)==0)/len(rule_matches_z)


print(rule_matches_z.shape)
print(mapping_rules_labels_t.shape)
print(train_X.shape)
print(train_X_finetuned.shape)
print(test_X_finetuned.shape)
print(len(gold_labels_test))
NUM_OUTPUT_CLASSES = 14
'''
model = LogisticRegressionModel(train_X.shape[1], NUM_OUTPUT_CLASSES)


custom_model_config = MajorityConfig(
    #optimizer=AdamW,
    lr=1e-4, # try 1e-3,1e-2
    batch_size=64, # try 16, 32 ,..., 256,
    epochs=50, # try 50 and 100
    filter_non_labelled=True, # try false
    #other_class_id = 8, # no finding
    output_classes = 14,
    multi_label = True,
    multi_label_threshold=0.001,
    criterion = BCEWithLogitsLoss
)
# config weigths tensor of weigths reduce weight for other class
trainer = MajorityVoteTrainer(
  model=model,
  mapping_rules_labels_t=mapping_rules_labels_t,
  model_input_x=TensorDataset(torch.from_numpy(train_X)),
  rule_matches_z=rule_matches_z,
  trainer_config=custom_model_config

)

trainer.train()

eval_dict, _ = trainer.test(TensorDataset(torch.from_numpy(test_X_finetuned)), gold_labels_test)

print(f"Accuracy: {eval_dict.get('accuracy')}")
'''
# Auto trainer

import os
from torch import Tensor
from tqdm.auto import tqdm

import joblib
#from minio import Minio

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

# initilize model
logreg_model = LogisticRegressionModel(train_X.shape[1], 14)

configs = [
    MajorityConfig(optimizer=AdamW,
                   lr=1e-4,
                   batch_size=16,
                   epochs=3,
                   output_classes = 14,
                   multi_label=True,
                   multi_label_threshold = 0.5,
                   criterion = BCEWithLogitsLoss)
]


trainer = MultiTrainer(
    name=["majority"],
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=TensorDataset(torch.from_numpy(train_X)),
    rule_matches_z=rule_matches_z,
    trainer_config=configs
)

# Run training
trainer.train()

# Run evaluation
metrics = trainer.test(TensorDataset(torch.from_numpy(test_X_finetuned)), gold_labels_test)
for trainer, metric in metrics.items():
    print(f"Trainer: {trainer}, accuracy: {metric[0].get('accuracy')}")