from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer


def convert_text_to_transformer_input(tokenizer, texts: List[str]) -> TensorDataset:
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    input_values_x = TensorDataset(input_ids, attention_mask)

    return input_values_x


def create_bert_input(
        model: str, train_x: List[str], rule_matches_z: np.array, mapping_rules_labels_t: np.array, train_y: np.array,
        test_x: List[str], test_y: np.array
) -> [TensorDataset, np.array, np.array, TensorDataset, TensorDataset, TensorDataset]:
    """Note that this can also be used for DistillBert and other versions.

    :param tokenizer: An aribrary tokenizer from the transformers library.
    :return: Data relevant for BERT training.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_ids_mask = convert_text_to_transformer_input(tokenizer, train_x)
    train_rule_matches_z = rule_matches_z
    if train_y is not None:
        train_y = TensorDataset(torch.from_numpy(train_y))

    test_ids_mask = convert_text_to_transformer_input(tokenizer, test_x)
    test_y = TensorDataset(torch.from_numpy(test_y))

    return (
        train_ids_mask, train_rule_matches_z, mapping_rules_labels_t, train_y,
        test_ids_mask, test_y
    )
