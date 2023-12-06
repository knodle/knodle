import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from examples.data_preprocessing.atis_dataset.atis_dataset_preprocessing import read_atis
from knodle.model.lstm_model import LSTMModel
from knodle.trainer import MajorityConfig
from knodle.trainer.baseline.majority import MajorityVoteSeqTrainer
from knodle.trainer.baseline.utils import accuracy_padded
from knodle.transformation.majority import seq_input_to_majority_vote_input


def test_01_majority_vote():
    no_lf = [0, 0, 0, 0]
    lf_1a = [1, 0, 0, 0]
    # lf_1b = [0, 1, 0, 0]
    lf_1ab = [1, 1, 0, 0]
    lf_2a = [0, 0, 1, 0]
    lf_2b = [0, 0, 0, 1]
    all_lfs = [1, 1, 1, 1]
    other_label = [1, 0, 0]
    label_1 = [0, 1, 0]
    label_2 = [0, 0, 1]
    label_1_2 = [0, 0.5, 0.5]

    rule_matches_z = np.array([[lf_1a, lf_2a, all_lfs], [lf_2b, lf_1ab, no_lf]])
    expected_labels = np.array([[label_1, label_2, label_1_2], [label_2, label_1, other_label]])
    mapping_rules_labels_t = np.array([label_1, label_1, label_2, label_2])
    voted_labels = seq_input_to_majority_vote_input(rule_matches_z, mapping_rules_labels_t, other_class_id=0)
    assert voted_labels.tolist() == expected_labels.tolist()


# def test_02_atis_majority_acc():
#     train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, t_matrix, id_to_word, id_to_lf, \
#         id_to_label = read_atis()
#
#     majority_acc = float(
#         accuracy_padded(
#             torch.tensor(np.zeros_like(dev_labels_padded)),
#             torch.tensor(dev_labels_padded),
#             mask=torch.tensor(dev_sents_padded.tensors[0] != 0)
#         )
#     )
#
#     assert 0.63 == pytest.approx(majority_acc, rel=1e-2)


# def test_03_atis_untrained_acc():
#     train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, t_matrix, id_to_word, id_to_lf, \
#         id_to_label = read_atis()
#
#     model = LSTMModel(vocab_size=len(id_to_word), tagset_size=len(id_to_label), embedding_dim=15, hidden_dim=15)
#
#     predicted_labels = model(dev_sents_padded.tensors[0]).argmax(axis=2)
#
#     majority_acc = float(
#         accuracy_padded(
#             torch.tensor(np.zeros_like(dev_labels_padded)),
#             torch.tensor(dev_labels_padded),
#             mask=torch.tensor(dev_sents_padded.tensors[0] != 0)
#         )
#     )
#
#     untrained_acc = float(
#         accuracy_padded(
#             predicted_labels,
#             dev_sents_padded.tensors[0],
#             mask=torch.tensor(dev_sents_padded.tensors[0] != 0))
#     )
#     assert untrained_acc <= majority_acc + 0.05


# def test_04_train_evaluate_atis():
#     train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, t_matrix, id_to_word, id_to_lf, \
#         id_to_label = read_atis()
#
#     model = LSTMModel(vocab_size=len(id_to_word), tagset_size=len(id_to_label), embedding_dim=15, hidden_dim=15)
#
#     trainer = MajorityVoteSeqTrainer(
#         model=model,
#         mapping_rules_labels_t=t_matrix,
#         trainer_config=MajorityConfig(epochs=10)
#     )
#     trainer.train(
#         model_input_x=train_sents_padded,
#         rule_matches_z=train_lfs_padded,
#         dev_model_input_x=dev_sents_padded,
#         dev_gold_labels_y=dev_labels_padded,
#     )
#
#     predicted_labels = trainer.model(torch.tensor(dev_sents_padded)).argmax(axis=2)
#     trained_acc = float(accuracy_padded(
#         predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0))
#     )
#     assert trained_acc >= 0.85
