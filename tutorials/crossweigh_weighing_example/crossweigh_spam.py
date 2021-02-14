import logging
import argparse
import os
import sys
from itertools import product

import scipy
import torch
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeigh
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from tutorials.crossweigh_weighing_example import utils

NUM_CLASSES = 2
logger = logging.getLogger(__name__)


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

    train_df, test_df, rule_matches_z, test_rule_matches_z, mapping_rules_labels_t, y_test = read_train_dev_test(
        path_to_data
    )

    train_tfidf_sparse, test_tfidf_sparse = create_tfidf_values(
        train_df.text.values,
        test_df.text.values
    )

    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset = TensorDataset(train_tfidf)

    test_tfidf = Tensor(test_tfidf_sparse.toarray())
    test_dataset = TensorDataset(test_tfidf)
    test_labels = TensorDataset(Tensor(y_test))

    parameters = dict(
        # use_weights=[True],
        lr=[1e-4],  # 0.1, 0.8, 1.0
        cw_lr=[0.0],  # 0.01,
        epochs=[2],  # 25, 35, 50, 100
        cw_partitions=[0],
        cw_folds=[0],  # 7,
        cw_epochs=[0],  # 1,
        weight_reducing_rate=[0],  # 0.5, 0.7
        samples_start_weights=[0],  # 2.0, 3.0, 4.0
    )
    param_values = [v for v in parameters.values()]

    tb = SummaryWriter('runs')

    for run_id, (lr, cw_lr, epochs, cw_part, cw_folds, cw_epochs, weight_rr, start_weights) in \
            enumerate(product(*param_values)):
        comment = f' lr = {lr} cw_lr = {cw_lr} cw_partitions = {cw_part} cw_folds = {cw_folds} ' \
                  f'cw_epochs = {cw_epochs} weight_reducing_rate = {weight_rr} samples_start_weights {start_weights}'

        print("Parameters: {}".format(comment))
        tb = SummaryWriter(comment=comment)

        folder_prefix = f'{cw_lr}_{cw_part}_{cw_folds}_{cw_epochs}_{weight_rr}_{start_weights}'
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
            optimizer_=torch.optim.Adam(model.parameters(), lr=cw_lr),
            output_classes=NUM_CLASSES,
            filter_empty_probs=True
        )
        custom_crossweigh_trainer_config = CrossWeighTrainerConfig(
            model=model,
            output_classes=NUM_CLASSES,
            optimizer_=torch.optim.Adam(model.parameters(), lr=lr),
            epochs=epochs,
            filter_empty_probs=True
        )

        trainer = CrossWeigh(
            model=model,
            rule_assignments_t=mapping_rules_labels_t,
            inputs_x=train_dataset,
            rule_matches_z=rule_matches_z,
            path_to_weights=path_to_weights,
            denoising_config=custom_crossweigh_denoising_config,
            trainer_config=custom_crossweigh_trainer_config,
            use_weights=False,
            run_classifier=True
        )
        trainer.train()
        clf_report = trainer.test(test_dataset, test_labels)
        logger.info(clf_report)

        tb.add_hparams(
            {"lr": lr,
             "cw_lr": cw_lr,
             "epochs": epochs,
             "cw_partitions": cw_part,
             "cw_folds": cw_folds,
             "cw_epochs": cw_epochs,
             "weight_reducing_rate": weight_rr,
             "samples_start_weights": start_weights},
            {"macro_avg_precision": clf_report["macro avg"]["precision"],
             "macro_avg_recall": clf_report["macro avg"]["recall"],
             "macro_avg_f1": clf_report["macro avg"]["f1-score"]}
        )

        tb.close()
        print("========================================================================")
        print("========================================================================")
        print("========================== RUN {} IS DONE ==============================".format(run_id))
        print("========================================================================")


def create_tfidf_values(train_data: [str], test_data: [str]):
    vectorizer = TfidfVectorizer()
    train_transformed_data = vectorizer.fit_transform(train_data)
    test_transformed_data = vectorizer.transform(test_data)
    # dump(transformed_data, "/Users/asedova/PycharmProjects/knodle/tutorials/conll_relation_extraction_dataset/output_data_with_added_arg_types/tfidf.lib")
    return train_transformed_data, test_transformed_data


def read_train_dev_test(target_path: str):
    train_df = load(os.path.join(target_path, 'df_train.lib'))
    test_df = load(os.path.join(target_path, 'df_test.lib'))
    train_rule_matches_z = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
    test_rule_matches_z = load(os.path.join(target_path, 'test_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(target_path, 'mapping_rules_labels.lib'))
    y_test = load(os.path.join(target_path, 'Y_test.lib'))

    return train_df, test_df, train_rule_matches_z, test_rule_matches_z, mapping_rules_labels_t, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--sample_weights", help="If there are pretrained samples sample_weights")

    args = parser.parse_args()

    train_crossweigh(args.path_to_data,
                     args.sample_weights)
