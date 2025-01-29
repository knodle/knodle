import os
import pandas as pd

from typing import Optional, Callable, Any

import joblib

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningDataModule


class KnodleDataModule(LightningDataModule):
    def __init__(
            self, data_dir: str = None,
            minio_dir: str = None,
            batch_size: int = 32,
            preprocessing_fct: Callable[[Any], np.ndarray] = lambda x, y: np.array([0, 0]),
            preprocessing_kwargs={"max_features": 400},
            dataloader_train_keys=[""]

    ):
        super().__init__()

        self.data_dir = data_dir
        self.minio_dir = minio_dir

        self.batch_size = batch_size

        self.preprocessing_fct = preprocessing_fct
        self.preprocessing_kwargs = preprocessing_kwargs

        self.data = {}

        self.dataloader_train_keys = dataloader_train_keys

    def setup(self, stage: Optional[str] = None) -> None:
        self.data["mapping_rules_labels_t"] = joblib.load(os.path.join(self.data_dir, "mapping_rules_labels_t.lib"))
        for split in ["train", "dev", "test"]:
            self.data[f"{split}_df"] = pd.read_csv(os.path.join(self.data_dir, f"df_{split}.csv"), sep=";")
            self.data[f"{split}_rule_matches_z"] = joblib.load(
                os.path.join(self.data_dir, f"{split}_rule_matches_z.pbz2"))

            if split != 'train':
                self.data[f"{split}_y"] = joblib.load(os.path.join(self.data_dir, f"{split}_labels.pbz2"))

        self.data = self.preprocessing_fct(self.data, **self.preprocessing_kwargs)

    def train_dataloader(self):
        data_loader_dict = {
            "z": DataLoader(self.data["train_rule_matches_z"], batch_size=self.batch_size)
        }
        for key in self.dataloader_train_keys:
            data_loader_dict[key] = DataLoader(self.data[f"train_{key}"], batch_size=self.batch_size)

        if "train_weak_y" in self.data:
            data_loader_dict["labels"] = DataLoader(self.data["train_weak_y"], batch_size=self.batch_size)

        return data_loader_dict

    def test_dataloader(self):
        arrays = []
        ds = TensorDataset(
            torch.from_numpy(self.data["test_x"]),
            torch.from_numpy(self.data["test_y"])
        )
        # data_loader_dict = {
        #     "x": DataLoader(np_array_to_tensor_dataset(self.data["test_x"]), batch_size=self.batch_size),
        #     "y": DataLoader(self.data["test_y"], batch_size=self.batch_size)
        # }
        return DataLoader(ds)
