import torch
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from examples.data_preprocessing.atis_dataset.atis_dataset_preprocessing import read_atis
from knodle.model.lstm_model import LSTMModel
from knodle.trainer import MajorityConfig
from knodle.trainer.baseline.majority import MajorityVoteSeqTrainer
from knodle.trainer.baseline.utils import accuracy_padded

train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, t_matrix, id_to_word, id_to_lf, id_to_label \
    = read_atis()

dev_sents_padded = TensorDataset(Tensor(dev_sents_padded))
dev_labels_padded = TensorDataset(Tensor(dev_labels_padded))

model = LSTMModel(vocab_size=len(id_to_word), tagset_size=len(id_to_label), embedding_dim=15, hidden_dim=15)

trainer = MajorityVoteSeqTrainer(
    model=model,
    mapping_rules_labels_t=t_matrix,
    trainer_config=MajorityConfig(epochs=10, criterion=CrossEntropyLoss, output_classes=48)
)

trainer.train(
    model_input_x=train_sents_padded,
    rule_matches_z=train_lfs_padded,    # #number of samples x #tokens in samples x #LFs
    dev_model_input_x=dev_sents_padded,
    dev_gold_labels_y=dev_labels_padded
)

predicted_labels = trainer.model(torch.tensor(dev_sents_padded)).argmax(axis=2)

print([(id_to_word[wid], id_to_label[lid]) for (wid, lid) in zip(dev_sents_padded[0], predicted_labels[0].tolist())])

# predicted_labels = model(torch.tensor(dev_sents_padded)).argmax(axis=2)
# print(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)))

dev_acc = float(
    accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0))
)

majority_acc = float(
    accuracy_padded(
        torch.tensor(np.zeros_like(dev_labels_padded)),
        torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)
    )
)

print(dev_acc)

print(majority_acc)
