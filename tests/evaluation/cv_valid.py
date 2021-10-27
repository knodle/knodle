import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import TensorDataset

from knodle.trainer.utils.validation_with_cv import get_val_cv_dataset


def test_get_val_cv_dataset():

    x = TensorDataset(Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                              [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]))
    z = np.array([[1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    labels = np.array([0, 0, 0, 1, 1])

    gold_dataset = [
        (Tensor([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]), Tensor([1, 0])),
        (Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]), Tensor([0, 1])),
        (Tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), Tensor([0, 1, 0, 0])),
        (Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]), Tensor([0, 1]))
    ]

    arr_train_datasets, arr_test_features, arr_test_labels = get_val_cv_dataset(x, z, labels, folds=4, seed=1234)

    assert torch.all(torch.eq(arr_test_features[0].tensors[0], gold_dataset[0][0]))
    assert torch.all(torch.eq(arr_test_labels[0].tensors[0], gold_dataset[0][1]))
    assert torch.all(torch.eq(arr_test_features[1].tensors[0], gold_dataset[1][0]))
    assert torch.all(torch.eq(arr_test_labels[1].tensors[0], gold_dataset[1][1]))
    assert torch.all(torch.eq(arr_test_features[2].tensors[0], gold_dataset[2][0]))
    assert torch.all(torch.eq(arr_test_labels[2].tensors[0], gold_dataset[2][1]))
    assert torch.all(torch.eq(arr_test_features[3].tensors[0], gold_dataset[3][0]))
    assert torch.all(torch.eq(arr_test_labels[3].tensors[0], gold_dataset[3][1]))
