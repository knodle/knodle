from torch.nn import BCEWithLogitsLoss
import numpy as np
from joblib import load
import torch
from torch.utils.data import TensorDataset
from transformers import AdamW
from knodle.trainer import MajorityConfig
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.multi_trainer import MultiTrainer


# load
Z = "rule_matches_z.lib"
T = "mapping_rules_labels_t.lib"
X_finetuned = "train_X_finetuned.lib"
X_test_finetuned = "X_test_finetuned.lib"
y_test = "y_gold_test.lib"

rule_matches_z = load(Z)
mapping_rules_labels_t = load(T)
train_X_finetuned = load(X_finetuned)
gold_labels_test = load(y_test)
test_X_finetuned = load(X_test_finetuned)
gold_labels_test = load(y_test)

# racall: number of studies without any matches
sum(np.sum(rule_matches_z, axis = 1)==0)/len(rule_matches_z)

# print shapes to get an overview of the data
print(rule_matches_z.shape)
print(mapping_rules_labels_t.shape)
print(train_X_finetuned.shape)
print(test_X_finetuned.shape)
print(len(gold_labels_test))
NUM_OUTPUT_CLASSES = 14

# init model
logreg_model = LogisticRegressionModel(train_X_finetuned.shape[1], NUM_OUTPUT_CLASSES)

configs = [
    MajorityConfig(optimizer=AdamW,
                   lr=1e-4,
                   batch_size=64,
                   epochs=100,
                   output_classes = NUM_OUTPUT_CLASSES,
                   multi_label=True,
                   multi_label_threshold = 0.5,
                   criterion = BCEWithLogitsLoss)
]

trainer = MultiTrainer(
    name=["majority"],
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=TensorDataset(torch.from_numpy(train_X_finetuned)),
    rule_matches_z=rule_matches_z,
    trainer_config=configs
)

# Run training
trainer.train()

# Run evaluation
metrics = trainer.test(TensorDataset(torch.from_numpy(test_X_finetuned)), gold_labels_test)
print(metrics)