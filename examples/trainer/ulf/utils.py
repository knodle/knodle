import pickle
import random
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from wrench.dataset import TextDataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def bert_text_extractor(
        data, path: str, split: str = None, cache_name: str = None, model_name: str = None, feature: str = "cls",
        device: torch.device = None
) -> np.array:
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path = Path(path)
    if cache_name is not None:
        curr_path = path / f'{split}_{cache_name}.pkl'
        if curr_path.exists():
            logger.info(f'loading features from {curr_path}')
            return pickle.load(open(curr_path, 'rb'))
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # e.g. 'bert-base-cased'
    text_features = []
    for sentence in tqdm(data):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, return_attention_mask=False,
                           return_token_type_ids=False)
        inputs = inputs['input_ids'].to(device)
        if feature == 'cls':
            output = model(inputs).pooler_output.cpu().squeeze().numpy()  # [len(sentence), dim_out]
            text_features.append(output)
        elif feature == 'avr':
            output = model(inputs).last_hidden_state.cpu().squeeze().numpy()  # [len(sentence), dim_out]
            text_features.append(np.average(output, axis=0))
    features = np.array(text_features)
    if cache_name is not None:
        path = path / f'{split}_{cache_name}.pkl'
        logger.info(f'saving features into {path}')
        pickle.dump(features, open(path, 'wb'), protocol=4)
    return features


def z_to_wrench_labels(z_matrix: np.array, t_matrix: np.array) -> np.array:
    lf2class = {}
    for row_id, row in enumerate(t_matrix):
        matched_class = list(np.where(row == 1)[0])
        if len(matched_class) != 1:
            matched_class = random.choice(matched_class)
        else:
            matched_class = int(matched_class[0])
        lf2class[row_id] = matched_class
    wrench_z = np.where(z_matrix == 0, -1, 1)
    w = np.where(wrench_z == 1)
    wrench_z[w] = [lf2class[lf] for lf in list(w[1])]
    return wrench_z.tolist()


class WrenchDataset(TextDataset):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.features: List = []
        self.ids: List = []
        self.labels: List = []
        self.examples: List = []
        self.weak_labels: List[List] = []
        self.id2label = {}

        self.load(**kwargs)

    def load(self, **kwargs):
        params = kwargs.keys()
        if "features" in params:
            self.features = kwargs["features"]
            self.ids = list(range(0, len(self.features)))

        if "weak_labels" in params:
            self.weak_labels = kwargs["weak_labels"]
            self.labels = [-100] * len(self.weak_labels)  # fake labels if non is given (i.e. for training set)

        if "labels" in params:
            self.labels = kwargs["labels"]

        if "examples" in params:
            self.examples = kwargs["examples"]

        if "n_class" in params:
            self.n_class = kwargs["n_class"]

        if "n_lf" in params:
            self.n_lf = kwargs["n_lf"]

        if "id2label" in params:
            self.id2label = kwargs["id2label"]

        if "ids" in params:
            self.ids = kwargs["ids"]

