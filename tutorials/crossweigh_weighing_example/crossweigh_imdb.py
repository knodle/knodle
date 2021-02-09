import argparse
import os
import sys

import scipy
import torch
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from torch import Tensor
from torchtext.vocab import GloVe
from torch.utils.tensorboard import SummaryWriter

from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.model.logistic_regression.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeigh
from tutorials.ImdbDataset.utils import read_train_dev_test
from tutorials.crossweigh_weighing_example import utils
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
# from tutorials.crossweigh_weighing_example.utils import encode_samples
from itertools import product

# from tutorials.knn_tfidf_similarity_example.knn_sim_tutorial_with_tacred import read_evaluation_data

NUM_CLASSES = 2


def train_crossweigh(
        path_to_data: str,
        path_sample_weights: str = None,
) -> None:
    """
    Training the model with CrossWeigh model denoising
    :param path_train_samples: path to DataFrame with training data
    :param path_z: path to binary matrix that contains info about rules matched in samples (samples x rules)
    :param path_t: path to binary matrix that contains info about which rule corresponds to which label (rule x labels)
    :param path_dev_features_labels: path to DataFrame with development data (1st column - samples, 2nd column - labels)
    :param path_word_emb_file: path to file with pretrained embeddings
    """

    train_df, dev_df, test_df, rule_matches_z, _, _, imdb_dataset, rule_assignments_t = read_train_dev_test(
        path_to_data)

    # train_data = pd.read_csv(path_train_samples)
    # rule_matches_z = load(path_z)
    rule_matches_z = rule_matches_z.toarray() if scipy.sparse.issparse(rule_matches_z) else rule_matches_z

    mapping_rules_labels_t = rule_assignments_t
    # dev_data = load(path_dev_features_labels)
    # test_data = load(path_test_features_labels)

    train_tfidf_sparse, dev_tfidf_sparse, test_tfidf_sparse = create_tfidf_values(
        train_df.reviews_preprocessed.values,
        dev_df.reviews_preprocessed.values,
        test_df.reviews_preprocessed.values
    )

    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset = TensorDataset(train_tfidf)

    test_tfidf = Tensor(test_tfidf_sparse.toarray())
    test_dataset = TensorDataset(test_tfidf)
    test_labels = torch.LongTensor(test_df.label_id.values)

    dev_tfidf = Tensor(dev_tfidf_sparse.toarray())
    dev_dataset = TensorDataset(dev_tfidf)
    dev_labels = torch.LongTensor(dev_df.label_id.values)

    parameters = dict(
        use_weights=[True],
        lr=[0.1, 2.0, 0.8],  # 0.1, 0.8, 1.0
        cw_lr=[0.8],  # 0.01,
        epochs=[1],  # 25, 35, 50, 100
        cw_partitions=[1],
        cw_folds=[3],  # 7,
        cw_epochs=[1],  # 1,
        weight_reducing_rate=[0.7],  # 0.5, 0.7
        samples_start_weights=[3.0],  # 2.0, 3.0, 4.0
    )
    param_values = [v for v in parameters.values()]

    tb = SummaryWriter('runs_baselines')

    for run_id, (use_weights, lr, cw_lr, epochs, cw_part, cw_folds, cw_epochs, weight_rr, start_weights) in \
            enumerate(product(*param_values)):
        comment = f' lr = {lr} cw_lr = {cw_lr} epochs = {epochs} cw_partitions = {cw_part} cw_folds = {cw_folds} ' \
                  f'cw_epochs = {cw_epochs} weight_reducing_rate = {weight_rr} samples_start_weights {start_weights} ' \
                  f'use_weights = {use_weights}'

        print("Parameters: {}".format(comment))
        tb = SummaryWriter(comment=comment)

        folder_prefix = f'{cw_lr}_{cw_part}_{cw_folds}_{cw_epochs}_{weight_rr}_{start_weights}_{use_weights}'
        print("Weights will be saved to/loaded from {}".format(folder_prefix))

        path_to_weights = os.path.join(path_sample_weights, folder_prefix)

        model = LogisticRegressionModel(train_tfidf.shape[1], NUM_CLASSES)

        custom_crossweigh_denoising_config = CrossWeighDenoisingConfig(
            model=model,
            crossweigh_partitions=cw_part,
            crossweigh_folds=cw_folds,
            crossweigh_epochs=cw_epochs,
            weight_reducing_rate=weight_rr,
            samples_start_weights=start_weights,
            lr=cw_lr,
            optimizer_=torch.optim.Adam(model.parameters()),
            output_classes=NUM_CLASSES,
            filter_empty_probs=True
        )
        custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
            model=model,
            lr=lr,
            output_classes=NUM_CLASSES,
            optimizer_=torch.optim.Adam(model.parameters()),
            epochs=epochs,
            filter_empty_probs=True
        )

        trainer = CrossWeigh(
            model=model,
            rule_assignments_t=mapping_rules_labels_t,
            inputs_x=train_dataset,
            rule_matches_z=rule_matches_z,
            dev_features=dev_dataset,
            dev_labels=dev_labels,
            path_to_weights=path_to_weights,
            denoising_config=custom_crossweigh_denoising_config,
            trainer_config=custom_crossweigh_trainer_config,
            run_classifier=True,
            use_weights=use_weights
        )
        trainer.train()
        clf_report = trainer.test(test_dataset, TensorDataset(test_labels))

        tb.add_hparams(
            {"lr": lr,
             "cw_lr": cw_lr,
             "epochs": epochs,
             "cw_partitions": cw_part,
             "cw_folds": cw_folds,
             "cw_epochs": cw_epochs,
             "weight_reducing_rate": weight_rr,
             "samples_start_weights": start_weights,
             "use_sample_weights": use_weights},
            {"macro_avg_precision": clf_report["macro avg"]["precision"],
             "macro_avg_recall": clf_report["macro avg"]["recall"],
             "macro_avg_f1": clf_report["macro avg"]["f1-score"]}
        )

        tb.close()
        print("========================================================================")
        print("========================================================================")
        print("========================== RUN {} IS DONE ==============================".format(run_id))
        print("========================================================================")


def create_tfidf_values(train_data: [str], dev_data: [str], test_data: [str]):
    # if os.path.exists("/Users/asedova/PycharmProjects/knodle/tutorials/ImdbDataset/data/tfidf.lib"):
    #     cached_data = load("/Users/asedova/PycharmProjects/knodle/tutorials/ImdbDataset/data/tfidf.lib")
    #     if cached_data.shape == text_data.shape:
    #         return cached_data

    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(train_data)
    dev_transformed_data = vectorizer.transform(dev_data)
    test_transformed_data = vectorizer.transform(test_data)
    # dump(transformed_data, "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_with_added_arg_types/tfidf.lib")
    return train_transformed_data, dev_transformed_data, test_transformed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--sample_weights", help="If there are pretrained samples sample_weights")

    args = parser.parse_args()

    train_crossweigh(args.path_to_data,
                     args.sample_weights)
