import numpy as np
from numpy.testing import assert_array_equal

from torch import Tensor, equal
from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset


def test_input_labels_to_tensordataset():

    a = np.ones((4, 4))
    b = np.ones((4, 3))
    labels = np.ones((4,))

    input_data = TensorDataset(Tensor(a), Tensor(b))

    input_label_dataset = input_labels_to_tensordataset(input_data, labels)

    assert len(input_label_dataset.tensors) == 3
    assert_array_equal(input_label_dataset.tensors[-1].cpu().detach().numpy(), labels)