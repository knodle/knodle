from typing import List, Union, Tuple, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pytorch_lightning import LightningModule
from transformers import AutoTokenizer


def get_tfidf_features(
        data, max_features: int = None
) -> Dict:

    vectorizer = TfidfVectorizer(max_features=max_features)

    data["train_x"] = vectorizer.fit_transform(data["train_df"]["sample"]).toarray()
    if data["test_df"] is not None:
        data["test_x"] = vectorizer.transform(data["test_df"]["sample"]).toarray()
    if data["dev_df"] is not None:
        data["dev_x"] = vectorizer.transform(data["dev_df"]["sample"]).toarray()

    return data


def get_transformer_features(
        data, transformer_name: str = 'distilbert-base-uncased', max_length: int = 512
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:

    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    splits = ["train", "test"]

    for split in splits:
        tokenized = tokenizer(data[f"{split}_df"]["sample"].tolist(), max_length=max_length, truncation=True, padding=True)
        data[f"{split}_input_ids"] = np.array(tokenized["input_ids"])
        data[f"{split}_attention_mask"] = np.array(tokenized["attention_mask"])

    return data
