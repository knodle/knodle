import argparse
import os
import sys
from typing import Union, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor, LongTensor
from torch.optim import SGD
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.crossweigh_weighing.config import DSCrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.dscrossweigh import DSCrossWeighTrainer
from tutorials.utils import get_samples_list, read_train_dev_test


def train_crossweigh(path_to_data: str, num_classes: int) -> None:
    """
    We are going to train a BERT classification model using weakly annotated data with additional DSCrossWeigh
    denoising. The sample weights in DSCrossWeigh will be trained with logistic regression in order to, firstly,
    reduce the computational effort, and, secondly, demonstrate the ability of the algorithm to use different models
    for data denoising and classifier training.
    :param path_to_data: path to the folder where all the input data is stored
    :param num_classes: number of output classes
    """

    num_classes = int(num_classes)

    # first, the data is read from the file
    train_df, dev_df, test_df, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels = \
        read_train_dev_test(path_to_data, if_dev_data=True)

    # we calculate sample weights using logistic regression model (with TF-IDF features) and use the BERT model for final classifier training.
    train_tfidf_sparse, _ , _ = get_tfidf_features(train_df, 0)
    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_tfidf_dataset = TensorDataset(train_tfidf)

    # For the BERT training we convert train, dev and test data to BERT encoded features (namely, input indices and attention mask)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_bert_dataset = get_bert_encoded_features(train_df, tokenizer, 0)
    test_labels = TensorDataset(LongTensor(list(test_df.iloc[:, 1])))
    test_bert_dataset = get_bert_encoded_features(test_df, tokenizer, 0)

    # for some datasets dev set is not provided. If it is not the case, is would be encoded with BERT features
    # (since it is used only for final training) and passed to a class constructor or to train function
    if dev_df is not None:
        dev_dataset_bert = get_bert_encoded_features(dev_df, tokenizer, 0)
        dev_labels = TensorDataset(LongTensor(list(dev_df.iloc[:, 1])))
    else:
        dev_labels, dev_dataset_bert = None, None

    # define the all needed parameters in a dictionary for convenience (can also be directly passed to Trainer/Config)
    parameters = {
        "lr": 1e-4, "cw_lr": 0.8, "epochs": 5, "cw_partitions": 2, "cw_folds": 5, "cw_epochs": 2, "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }

    # define LogReg and BERT models for training sample weights and final classifier correspondingly
    cw_model = LogisticRegressionModel(train_tfidf.shape[1], num_classes)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    # define a custom DSCrossWeigh config. If no custom config is defined, the DSCrossWeighTrainer will use the default
    # DSCrossWeighDenoisingConfig which is stored in the fold with the DSCrossWeigh trainer
    custom_crossweigh_config = DSCrossWeighDenoisingConfig(
        # general trainer parameters
        output_classes=num_classes,
        filter_non_labelled=False,
        other_class_id=2,
        seed=12345,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=AdamW,
        lr=parameters.get("lr"),
        grad_clipping=5,
        caching=True,           # the sample weights will be saved
        caching_suffix=f"{parameters.get('cw_partitions')}_{parameters.get('cw_folds')}",
        output_dir_path=path_to_data,

        # dscrossweigh specific parameters
        partitions=parameters.get("cw_partitions"),  # number of dscrossweigh iterations (= splitting into folds)
        folds=parameters.get("cw_folds"),  # number of folds train data will be splitted into
        weight_reducing_rate=parameters.get("weight_rr"),  # sample weights reducing coefficient
        samples_start_weights=parameters.get("samples_start_weights"),  # the start weight of sample weights
        cw_epochs=parameters.get("cw_epochs"),  # number of epochs each dscrossweigh model is to be trained
        cw_optimizer=SGD,
        cw_lr=parameters.get("cw_lr")
    )

    trainer = DSCrossWeighTrainer(
        # general Trainer inputs (a more detailed explanation of Knodle inputs is in README)
        model=model,  # classification model
        mapping_rules_labels_t=t_mapping_rules_labels,  # t matrix
        model_input_x=train_bert_dataset,  # x matrix for training the classifier
        rule_matches_z=z_train_rule_matches,  # z matrix
        trainer_config=custom_crossweigh_config,

        # additional dev set used for classification model evaluation during training
        dev_model_input_x=dev_dataset_bert,
        dev_gold_labels_y=dev_labels,

        # dscrossweigh specific parameters. If they are not defined, the corresponding main classification parameters
        # will be used instead (model instead of cw_model etc)
        cw_model=cw_model,  # model that will be used for dscrossweigh weights calculation
        cw_model_input_x=train_tfidf_dataset,  # x matrix for training the dscrossweigh models
    )

    # the DSCrossWeighTrainer is trained
    trainer.train()
    # the trained model is tested on the test set
    clf_report, _ = trainer.test(test_bert_dataset, test_labels)
    print(clf_report)


def get_bert_encoded_features(
        input_data: Union[pd.Series, pd.DataFrame], tokenizer: DistilBertTokenizer, column_num: int = None
) -> TensorDataset:
    """
    Convert input data to BERT encoded features (more details about DistilBert Model could be found at
    https://huggingface.co/transformers/model_doc/distilbert.html)
    :param input_data: training/dev/test samples that are to be encoded with BERT features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param tokenizer: DistilBertTokenizer tokenizer for english from HuggingFace
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples are stored
    :return: TensorDataset with encoded data
    """
    encoding = tokenizer(get_samples_list(input_data, column_num), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return TensorDataset(input_ids, attention_mask)


def get_tfidf_features(
        train_data: Union[pd.DataFrame, pd.Series], column_num: int = None,
        test_data: Union[pd.DataFrame, pd.Series] = None, dev_data: Union[pd.DataFrame, pd.Series] = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """
    Convert input data to a matrix of TF-IDF features.
    :param train_data: training samples that are to be encoded with TF-IDF features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples
    are stored
    :param test_data: if DataFrame/Series with test data is provided
    :param dev_data: if DataFrame/Series with development data is provided, it will be encoded as well
    :return: TensorDataset with encoded data
    """
    dev_transformed_data, test_transformed_data = None, None
    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(get_samples_list(train_data, column_num))

    if test_data is not None:
        test_transformed_data = vectorizer.transform(get_samples_list(test_data, column_num))
    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(get_samples_list(dev_data, column_num))

    return train_transformed_data, test_transformed_data, dev_transformed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="Path to the folder where all input files are stored.")
    parser.add_argument("--num_classes", help="Number of classes")
    args = parser.parse_args()

    train_crossweigh(args.path_to_data, args.num_classes)
