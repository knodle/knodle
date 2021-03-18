import argparse
import os
import sys
from typing import Union, Tuple

import pandas as pd
import numpy as np
import torch
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.crossweigh_weighing.config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeighTrainer

NUM_CLASSES = 2
CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])


def train_crossweigh(
        path_to_data: str,
        path_sample_weights: str = None,
) -> None:
    """ Training the BERT model with data denoising using DSCrossWeigh algorithm with logistic regression model """

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_df, dev_df, test_df, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels = \
        read_train_dev_test(path_to_data, if_dev_data=True)

    train_tfidf_sparse, test_tfidf_sparse, dev_tfidf_sparse = get_tfidf_features(train_df, test_df, dev_df)
    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset = TensorDataset(train_tfidf)
    train_input_x_bert = get_bert_encoded_features(train_df, tokenizer)

    if dev_df is not None:
        dev_labels = TensorDataset(LongTensor(list(dev_df.iloc[:, 1])))
        dev_dataset_bert = get_bert_encoded_features(dev_df, tokenizer)
    else:
        dev_labels, dev_dataset_bert = None, None

    test_labels = TensorDataset(LongTensor(list(test_df.iloc[:, 1])))
    test_dataset_bert = get_bert_encoded_features(test_df, tokenizer)

    os.makedirs(path_sample_weights, exist_ok=True)

    parameters = {
        "lr": 1e-4,
        "cw_lr": 0.8,
        "epochs": 5,
        "cw_partitions": 2,
        "cw_folds": 5,
        "cw_epochs": 2,
        "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_CLASSES)
    cw_model = LogisticRegressionModel(train_tfidf.shape[1], NUM_CLASSES)

    custom_crossweigh_config = CrossWeighDenoisingConfig(
        output_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS,
        filter_non_labelled=False,
        other_class_id=41,
        if_set_seed=True,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=AdamW(model.parameters(), lr=parameters.get("lr")),
        grad_clipping=5,
        partitions=parameters.get("cw_partitions"),
        folds=parameters.get("cw_folds"),
        weight_reducing_rate=parameters.get("weight_rr"),
        samples_start_weights=parameters.get("samples_start_weights"),
        cw_epochs=parameters.get("cw_epochs"),
        cw_optimizer=torch.optim.Adam(cw_model.parameters(), lr=parameters.get("cw_lr"))
    )

    trainer = CrossWeighTrainer(
        model=model,
        cw_model=cw_model,
        mapping_rules_labels_t=t_mapping_rules_labels,
        model_input_x=train_input_x_bert,
        cw_model_input_x=train_dataset,
        dev_model_input_x=dev_dataset_bert,
        dev_gold_labels_y=dev_labels,
        rule_matches_z=z_train_rule_matches,
        path_to_weights=path_sample_weights,
        trainer_config=custom_crossweigh_config,
        use_weights=True,
        run_classifier=True
    )
    trainer.train()
    clf_report, _ = trainer.test(test_dataset_bert, test_labels)
    print(clf_report)


def read_train_dev_test(
        target_path: str, if_dev_data: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
           Tuple[pd.DataFrame, None, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """This function loads the matrices as well as train, test (occasionally, also dev) data from corresponding files"""
    train_df = load(os.path.join(target_path, 'df_train.lib'))
    test_df = load(os.path.join(target_path, 'df_test.lib'))
    z_train_rule_matches = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
    z_test_rule_matches = load(os.path.join(target_path, 'test_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(target_path, 'mapping_rules_labels.lib'))

    if if_dev_data:
        dev_df = load(os.path.join(target_path, 'df_dev.lib'))
        return train_df, dev_df, test_df, z_train_rule_matches, z_test_rule_matches, mapping_rules_labels_t

    return train_df, None, test_df, z_train_rule_matches, z_test_rule_matches, mapping_rules_labels_t


def get_bert_encoded_features(
        input_data: pd.Series, tokenizer: DistilBertTokenizer, column_num: int = None
) -> TensorDataset:
    """ Convert input data to BERT encoded features """
    if isinstance(input_data, pd.Series):
        encoding = tokenizer(list(input_data), return_tensors='pt', padding=True, truncation=True)
    elif isinstance(input_data, pd.DataFrame) and column_num:
        encoding = tokenizer(list(input_data.iloc[:, column_num]), return_tensors='pt', padding=True, truncation=True)
    else:
        raise ValueError("Please pass input data either as a Series or a DataFrame with specified number where samples"
                         "are stored")
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return TensorDataset(input_ids, attention_mask)


def get_tfidf_features(
        train_data: Union[pd.DataFrame, pd.Series], test_data: pd.Series, column_num: int = None,
        dev_data: pd.Series = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """ Convert input data to a matrix of TF-IDF features """
    vectorizer = TfidfVectorizer()

    if isinstance(train_data, pd.Series):
        train_transformed_data = vectorizer.fit_transform(list(train_data))
    elif isinstance(train_data, pd.DataFrame) and column_num:
        train_transformed_data = vectorizer.fit_transform(list(train_data.iloc[:, column_num]))
    else:
        raise ValueError("Please pass input data either as a Series or a DataFrame with specified number where samples"
                         "are stored")

    if isinstance(test_data, pd.Series):
        test_transformed_data = vectorizer.transform(list(test_data))
    elif isinstance(test_data, pd.DataFrame) and column_num:
        test_transformed_data = vectorizer.transform(list(test_data.iloc[:, column_num]))
    else:
        raise ValueError("Please pass input data either as a Series or a DataFrame with specified number where samples"
                         "are stored")

    if dev_data is not None:
        if isinstance(dev_data, pd.Series):
            dev_transformed_data = vectorizer.transform(list(dev_data))
        elif isinstance(dev_data, pd.DataFrame) and column_num:
            dev_transformed_data = vectorizer.transform(list(dev_data.iloc[:, column_num]))
        else:
            raise ValueError(
                "Please pass input data either as a Series or a DataFrame with specified number where samples"
                "are stored")
        return train_transformed_data, test_transformed_data, dev_transformed_data
    return train_transformed_data, test_transformed_data, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--sample_weights", help="If there are pretrained samples sample_weights")

    args = parser.parse_args()

    train_crossweigh(
        args.path_to_data, args.sample_weights
    )
