from typing import Any, List, Optional, Union, Callable
import numpy as np
from torch.utils import data
from tqdm import tqdm


class CosineDataset(data.Dataset):
    def __init__(self, xs, labels, weak_label_mat=None):
        self.xs = xs
        self.labels = labels
        self.weak_label_mat = weak_label_mat

    def __len__(self):
        return len(self.labels)

    def get_covered_subset(self):
        assert self.weak_label_mat is not None, "we need information in weak_label_mat to get covered subset"
        idx = [i for i in range(len(self)) if np.any(np.array(self.weak_label_mat[i]) != -1)]
        no_idx = [i for i in range(len(self)) if np.all(np.array(self.weak_label_mat[i]) == -1)]
        if len(idx) == len(self):  # all are selected
            return self, None
        elif len(idx) == 0:  # nothing is selected
            return None, self
        else:
            return self.create_subset(idx), self.create_subset(no_idx)

    def create_subset(self, indices: List[int]):
        # create a subset by index
        assert self.labels != []
        selected_xs = {k: v[indices] for k, v in self.xs.items()}
        selected_labels = self.labels[indices]
        dataset = CosineDataset(selected_xs, selected_labels)
        return dataset

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.xs.items()}
        item['labels'] = self.labels[index]
        item['index'] = index
        return item