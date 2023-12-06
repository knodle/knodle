import torch
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from examples.data_preprocessing.atis_dataset.atis_dataset_preprocessing import read_atis
from knodle.model.lstm_model import LSTMModel
from knodle.trainer import MajorityConfig
from knodle.trainer.baseline.majority import MajorityVoteSeqTrainer
from knodle.trainer.baseline.utils import accuracy_padded

# read the atis dataset
# train_sents_added: training data samples; #training samples x #padded tokens in sample (atis: 3982 x 20)
# train_z_padded: training z matrix; #samples x #padded tokens in sample x #LFs (atis: 3982 x 20 x 126)
# dev_sents_padded: validation data samples; #samples x #padded tokens in sample (atis: 996 x 20)
# dev_labels_padded: validation labels; #samples x #padded tokens in sample (atis: 996 x 20)
# t_matrix: t matrix; #LFs x #classes (atis: 126 x 48)
train_sents_padded, train_z_padded, dev_sents_padded, dev_labels_padded, t_matrix, id_to_word, id_to_lf, id_to_label \
    = read_atis(pad_length=20)
num_labels = t_matrix.shape[1]

# transform X matrix, X_dev and Y_dev into TensorDataset
train_sents_padded = TensorDataset(Tensor(train_sents_padded).to(torch.int64))
dev_sents_padded = TensorDataset(Tensor(dev_sents_padded).to(torch.int64))
dev_labels_padded = TensorDataset(Tensor(dev_labels_padded).to(torch.int64))

# define a model which will be trained
model = LSTMModel(vocab_size=len(id_to_word), tagset_size=num_labels, embedding_dim=30, hidden_dim=50)

# initialize config
custom_seq_config = MajorityConfig(
    epochs=20, criterion=CrossEntropyLoss, optimizer=SGD, output_classes=num_labels, other_class_id=0, lr=0.015
)

# initialize trainer
trainer = MajorityVoteSeqTrainer(
    model=model,
    mapping_rules_labels_t=t_matrix,
    trainer_config=custom_seq_config
)

trainer.train(
    model_input_x=train_sents_padded,
    rule_matches_z=train_z_padded,
    dev_model_input_x=dev_sents_padded,
    dev_gold_labels_y=dev_labels_padded
)

predicted_labels = trainer.get_predicted_labels(dev_sents_padded)

# print([(id_to_word[wid], id_to_label[lid]) for (wid, lid) in zip(dev_sents_padded[0], predicted_labels[0].tolist())])

# print(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)))

dev_acc = float(
    # accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0))
    accuracy_padded(predicted_labels, dev_labels_padded.tensors[0], mask=(dev_sents_padded.tensors[0] != 0))
)

majority_acc = float(
    accuracy_padded(
        torch.tensor(np.zeros_like(dev_labels_padded.tensors[0])),
        dev_labels_padded.tensors[0],
        mask=(dev_sents_padded.tensors[0] != 0)
    )
)

print(dev_acc)

print(majority_acc)
