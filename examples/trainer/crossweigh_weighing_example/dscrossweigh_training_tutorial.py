import argparse
import os
import sys
from typing import Union, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

from examples.utils import get_samples_list, read_train_dev_test, create_bert_encoded_features
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.crossweigh_weighing.config import DSCrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh import DSCrossWeighTrainer

CLASS_WEIGHTS = torch.FloatTensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])


def train_crossweigh(path_to_data: str, path_sample_weights: str, num_classes: int) -> None:
    """ Training the BERT model with data denoising using DSCrossWeigh algorithm with logistic regression model """

    num_classes = int(num_classes)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_df, dev_df, test_df, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels = \
        read_train_dev_test(path_to_data, if_dev_data=True)

    train_tfidf_sparse, test_tfidf_sparse, dev_tfidf_sparse = get_tfidf_features(train_df, test_df, dev_df)
    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset = TensorDataset(train_tfidf)
    train_input_x_bert = create_bert_encoded_features(train_df, tokenizer)

    if dev_df is not None:
        dev_labels = TensorDataset(LongTensor(list(dev_df.iloc[:, 1])))
        dev_dataset_bert = create_bert_encoded_features(dev_df, tokenizer)
    else:
        dev_labels, dev_dataset_bert = None, None

    test_labels = TensorDataset(LongTensor(list(test_df.iloc[:, 1])))
    test_dataset_bert = create_bert_encoded_features(test_df, tokenizer)

    os.makedirs(path_sample_weights, exist_ok=True)

    parameters = {
        "lr": 1e-4, "cw_lr": 0.8, "epochs": 5, "cw_partitions": 2, "cw_folds": 5, "cw_epochs": 2, "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    cw_model = LogisticRegressionModel(train_tfidf.shape[1], num_classes)

    custom_crossweigh_config = DSCrossWeighDenoisingConfig(
        output_classes=num_classes,
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

    trainer = DSCrossWeighTrainer(
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


def get_tfidf_features(
        train_data: Union[pd.DataFrame, pd.Series], test_data: pd.Series, column_num: int = None,
        dev_data: pd.Series = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """ Convert input data to a matrix of TF-IDF features """
    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(get_samples_list(train_data, column_num))
    test_transformed_data = vectorizer.transform(get_samples_list(test_data, column_num))

    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(get_samples_list(dev_data, column_num))
        return train_transformed_data, test_transformed_data, dev_transformed_data
    return train_transformed_data, test_transformed_data, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="Path to the folder where all input files are stored.")
    parser.add_argument("--sample_weights", help="Path to the folder that either sample weights will be saved to or "
                                                 "will be loaded from")
    parser.add_argument("--num_classes", help="Number of classes")
    args = parser.parse_args()

    train_crossweigh(args.path_to_data, args.sample_weights, args.num_classes)
