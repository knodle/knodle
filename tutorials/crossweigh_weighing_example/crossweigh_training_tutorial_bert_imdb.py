import argparse
import os
import sys
import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, AdamW, AutoModelForSequenceClassification, BertForSequenceClassification, \
    DistilBertForSequenceClassification

from knodle.evaluation.tacred_metrics import score
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.bert.crossweigh_with_bert import CrossWeigh
from tutorials.ImdbDataset.utils import read_train_dev_test

NUM_CLASSES = 2
# MAXLEN = 50

# number of columns in input DataFrames with corresponding data
TRAIN_SAMPLES = 2       # in train DataFrame samples are stored in column with index 1
DEV_SAMPLES = 2
TEST_SAMPLES = 2
DEV_LABELS = 3
TEST_LABELS = 3

logger = logging.getLogger(__name__)


def train_crossweigh(
        path_to_data: str,
        path_sample_weights: str = None,
) -> None:

    os.makedirs(path_sample_weights, exist_ok=True)
    # path_sample_weights = os.path.join(path_sample_weights, "0.8_2_15_2_0.3_2.0_True")

    train_df, dev_df, test_df, rule_matches_z, _, _, _, rule_assignments_t = read_train_dev_test(
            path_to_data)

    train_input_x_bert, dev_dataset_bert, dev_labels_bert, test_dataset_bert, test_labels_bert = get_bert_data(
        train_df, dev_df, test_df
    )
    # train_input_x_tfidf, dev_dataset_tfidf, dev_labels_tfidf, test_dataset_tfidf, test_labels_tfidf = get_tfidf_data(
    #     train_df, dev_df, test_df
    # )

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
        model=model,
        optimizer_=AdamW(model.parameters(), lr=0.01),
        output_classes=NUM_CLASSES,
        filter_empty_probs=True
    )

    custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
        model=model,
        output_classes=NUM_CLASSES,
        batch_size=64,
        optimizer_=AdamW(model.parameters(), lr=1e-4),
        filter_empty_probs=True
    )

    trainer = CrossWeigh(
        model=model,
        rule_assignments_t=rule_assignments_t,
        inputs_x=train_input_x_bert,
        rule_matches_z=rule_matches_z,
        dev_features=dev_dataset_bert,
        dev_labels=dev_labels_bert,
        path_to_weights=path_sample_weights,
        denoising_config=custom_crossweigh_denoising_config,
        trainer_config=custom_crossweigh_trainer_config,
        use_weights=False
    )

    # trainer.train()

    model.load_state_dict(torch.load(
        "/Users/asedova/PycharmProjects/knodle/data/imdb/sample_weights/0.8_2_10_2_0.7_4.0/trained_models/model_epoch_1.pth"
    ))

    print("Model is loaded from 0.8_2_10_2_0.7_4.0/trained_models")
    print("Testing on the test dataset....")
    # metrics = test(test_dataset, TensorDataset(test_labels))["macro avg"]
    # metrics = test(model, custom_crossweigh_trainer_config.optimizer, test_dataset, TensorDataset(test_labels), "cpu")

    clf_report = trainer.test(test_dataset_bert, test_labels_bert)
    logger.info(clf_report)


def get_bert_data(
        train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, TensorDataset, TensorDataset]:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_input_x = get_features(train_df.iloc[:, TRAIN_SAMPLES], tokenizer)

    dev_dataset = get_features(dev_df.iloc[:, DEV_SAMPLES], tokenizer)
    dev_labels = TensorDataset(torch.LongTensor(list(dev_df.iloc[:, DEV_LABELS])))  # label_id
    test_dataset = get_features(test_df.iloc[:, TEST_SAMPLES], tokenizer)
    test_labels = TensorDataset(torch.LongTensor(list(test_df.iloc[:, TEST_LABELS])))

    return train_input_x, dev_dataset, dev_labels, test_dataset, test_labels


def get_tfidf_data(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame):

    train_tfidf_sparse, dev_tfidf_sparse, test_tfidf_sparse = create_tfidf_values(
        train_df.reviews_preprocessed.values,
        dev_df.reviews_preprocessed.values,
        test_df.reviews_preprocessed.values
    )

    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_input_x = TensorDataset(train_tfidf)

    test_tfidf = Tensor(test_tfidf_sparse.toarray())
    test_dataset = TensorDataset(test_tfidf)
    test_labels = torch.LongTensor(test_df.label_id.values)

    dev_tfidf = Tensor(dev_tfidf_sparse.toarray())
    dev_dataset = TensorDataset(dev_tfidf)
    dev_labels = torch.LongTensor(dev_df.label_id.values)

    return train_input_x, dev_dataset, dev_labels, test_dataset, test_labels

def create_tfidf_values(train_data: [str], dev_data: [str], test_data: [str]):
    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(train_data)
    dev_transformed_data = vectorizer.transform(dev_data)
    test_transformed_data = vectorizer.transform(test_data)
    return train_transformed_data, dev_transformed_data, test_transformed_data

# # todo: taken from Andi's PR
# def input_labels_to_tensordataset(model_input_x: TensorDataset, labels: np.ndarray):
#
#     model_tensors = model_input_x.tensors
#     input_label_dataset = TensorDataset(*model_tensors, torch.from_numpy(labels))
#
#     return input_label_dataset
#
#
# def _load_batch(batch, device):
#     input_batch = [inp.to(device) for inp in batch[0: -1]]
#     label_batch = batch[-1].to(device)
#
#     return input_batch, label_batch
#
#
# def _make_dataloader(
#         dataset: TensorDataset, batch_size: int = 32, shuffle: bool = True
# ) -> DataLoader:
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         drop_last=False,
#         shuffle=shuffle,
#     )
#     return dataloader
#
#
# def test(model, optimizer, features_dataset: TensorDataset, labels: TensorDataset, device):
#         feature_label_dataset = input_labels_to_tensordataset(features_dataset, labels.tensors[0].cpu().numpy())
#         feature_label_dataloader = _make_dataloader(feature_label_dataset, shuffle=False)
#
#         model.to(device)
#         model.eval()
#         predictions_list, label_list = [], []
#         # Loop over predictions
#         with torch.no_grad():
#             for batch in tqdm(feature_label_dataloader):
#                 input_batch, label_batch = _load_batch(batch, device)
#
#                 # forward pass
#                 optimizer.zero_grad()
#                 outputs = model(*input_batch)
#                 if isinstance(outputs, torch.Tensor):
#                     prediction_vals = outputs
#                 else:
#                     prediction_vals = outputs[0]
#
#                 # add predictions and labels
#                 predictions = np.argmax(prediction_vals.detach().cpu().numpy(), axis=-1)
#                 predictions_list.append(predictions)
#                 label_list.append(label_batch.detach().cpu().numpy())
#
#         # transform to correct format
#         predictions = np.squeeze(np.hstack(predictions_list))
#         gold_labels = np.squeeze(np.hstack(label_list))
#
#         clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)
#
#         return clf_report

# def _test(model, trainer, test_features: TensorDataset, test_labels: Tensor, labels2ids: Dict) -> Dict:
#     feature_labels_dataset = TensorDataset(test_features.tensors[0], test_labels)
#     feature_labels_dataloader = trainer._make_dataloader(feature_labels_dataset)
#
#     model.eval()
#     all_predictions, all_labels = torch.Tensor(), torch.Tensor()
#     for features, labels in feature_labels_dataloader:
#         outputs = model(features)
#         _, predicted = torch.max(outputs, 1)
#         all_predictions = torch.cat([all_predictions, predicted])
#         all_labels = torch.cat([all_labels, labels])
#
#     predictions_idx, test_labels_idx = (all_predictions.cpu().detach().type(torch.IntTensor).tolist(),
#                                         all_labels.cpu().detach().type(torch.IntTensor).tolist())
#
#     idx2labels = dict([(value, key) for key, value in labels2ids.items()])
#
#     predictions = [idx2labels[p] for p in predictions_idx]
#     test_labels = [idx2labels[p] for p in test_labels_idx]
#
#     clf_report = classification_report(y_true=test_labels, y_pred=predictions, output_dict=True)
#
#     return clf_report


def get_features(data: pd.Series, tokenizer: DistilBertTokenizer) -> TensorDataset:
    encoding = tokenizer(data.tolist(), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    features = TensorDataset(input_ids, attention_mask)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--sample_weights", help="")

    args = parser.parse_args()

    train_crossweigh(args.path_to_data,
                     args.sample_weights)
