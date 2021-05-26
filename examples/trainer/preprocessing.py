from typing import List, Union, Tuple
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from torch.utils.data import TensorDataset


def get_tfidf_features(
        train_data: List, test_data: List = None, dev_data: List = None, path_to_cache: str = None,
        max_features: int = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """
    Convert input data to a matrix of TF-IDF features.
    :param train_data: training samples that are to be encoded with TF-IDF features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param test_data: if DataFrame/Series with test data is provided
    :param dev_data: if DataFrame/Series with development data is provided, it will be encoded as well
    :param path_to_cache: a path to the folder where calculated cached TF-IDF values should be saved
    :param max_features: If not None, build a vocabulary that only consider the top max_features ordered by term
    frequency across the corpus.
    :return: TensorDataset with encoded data
    """

    dev_transformed_data, test_transformed_data = None, None
    vectorizer = TfidfVectorizer(max_features=max_features)

    train_transformed_data = vectorizer.fit_transform(train_data)
    if test_data is not None:
        test_transformed_data = vectorizer.transform(test_data)
    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(dev_data)

    if path_to_cache:
        dump(train_transformed_data, path_to_cache)
        dump(dev_transformed_data, path_to_cache)
        dump(test_transformed_data, path_to_cache)

    return train_transformed_data, test_transformed_data, dev_transformed_data


def convert_text_to_transformer_input(tokenizer, texts: List[str]) -> TensorDataset:
    """
    Convert input data to BERT encoded features (more details could be found at
    https://huggingface.co/transformers/model_doc)
    :param texts: training/dev/test samples that are to be encoded with BERT features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param tokenizer: DistilBertTokenizer tokenizer for english from HuggingFace
    :return: TensorDataset with encoded data
    """
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    input_values_x = TensorDataset(input_ids, attention_mask)

    return input_values_x
