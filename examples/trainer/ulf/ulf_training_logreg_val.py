import logging
import argparse
import json
import os
import sys
from itertools import product
from random import randint

import joblib
from torch import Tensor, LongTensor
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test, collect_res_statistics, print_metrics
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.ulf import UlfTrainer
from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10
    output_file = f"{output_file}_0_3"

    parameters = dict(
        # seed=None,
        use_prior=[False],
        p=[0.3],      #0.5, 0.3
        lr=[0.01],       #  0.01,
        cv_n_folds=[15],        # 3, 5, 8, 10,
        iterations=[1],
        psx_calculation_method=['signatures']       #, 'rules', 'random'],      # how the splitting into folds will be performed
    )
    parameter_values = [v for v in parameters.values()]

    df_train = pd.read_csv(os.path.join(path_to_data, "df_train.csv"))
    df_test = pd.read_csv(os.path.join(path_to_data, "df_test.csv"))
    df_dev = pd.read_csv(os.path.join(path_to_data, "df_dev.csv"))

    mapping_rules_labels_t = joblib.load(os.path.join(path_to_data, "mapping_rules_labels_t.lib"))

    train_rule_matches_z = joblib.load(os.path.join(path_to_data, "train_rule_matches_z.lib"))
    test_rule_matches_z = joblib.load(os.path.join(path_to_data, "test_rule_matches_z.lib"))
    # df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
    #     path_to_data, if_dev_data=True)

    train_input_x, test_input_x, dev_input_x = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"]
    )

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    dev_features_dataset = TensorDataset(Tensor(dev_input_x.toarray()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))

    # create dev labels dataset
    dev_labels = df_dev["label"].tolist()
    dev_labels_dataset = TensorDataset(LongTensor(dev_labels))

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1
    results, exp_signatures = [], []

    for run_id, (params) in enumerate(product(*parameter_values)):

        use_prior, p, lr, folds, iterations, psx_method = params

        p = None if use_prior else p
        params_dict = {
            'prior': use_prior, 'epochs': 20, 'p': p, 'lr': lr, 'folds': folds, 'iter': iterations, 'psx': psx_method
        }
        params_signature = str(params_dict)
        log_section(params_signature, logger)

        if params_signature in exp_signatures:
            logger.info("This experiment was already done; skip it.")
            continue

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted = \
            [], [], [], [], []

        for exp in range(0, num_experiments):

            model = LogisticRegressionModel(train_input_x.shape[1], num_classes)
            custom_cleanlab_config = UlfConfig(
                iterations=iterations,
                cv_n_folds=folds,
                psx_calculation_method=psx_method,
                filter_non_labelled=False,
                use_probabilistic_labels=False,
                use_prior=use_prior,
                p=p,
                output_classes=num_classes,
                criterion=CrossEntropyLoss,
                epochs=20,
                grad_clipping=5,
                save_model_name=output_file,
                save_model_path="trained_models",
                optimizer=Adam,
                lr=lr,
                batch_size=256,
                early_stopping=True
            )
            logger.info(custom_cleanlab_config)
            trainer = UlfTrainer(
                model=model,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=train_features_dataset,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config,

                dev_model_input_x=dev_features_dataset,
                dev_gold_labels_y=dev_labels_dataset,

                df_train=df_train,
                output_file=os.path.join(
                    path_to_data, f"{output_file}_{exp}_{use_prior}_{p}_{folds}_{iterations}_{psx_method}"
                )
            )

            trainer.train()
            clf_report = trainer.test(test_features_dataset, test_labels_dataset)
            print_metrics(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1_avg.append(clf_report['macro avg']['f1-score'])
            exp_results_f1_weighted.append(clf_report['weighted avg']['f1-score'])

            # os.remove(
            #     "/Users/asedova/PycharmProjects/01_knodle/examples/trainer/ulf/trained_models/"
            #     "sms_ulf_logreg_10exp_best.pt"
            # )
            # os.remove(
            #     "/Users/asedova/PycharmProjects/01_knodle/examples/trainer/ulf/trained_models/sms_ulf_logreg_10exp_fin.pt"
            # )

        exp_signatures.append(params_signature)

        res = collect_res_statistics(
                exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted,
                params=params_dict, verbose=True
            )
        with open(os.path.join(path_to_data, f"{output_file}.json"), 'a+') as file:
            json.dump(res, file)

    #     results.append(
    #         collect_res_statistics(
    #             exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted,
    #             params=params_dict, verbose=True
    #         )
    #     )
    #
    # with open(os.path.join(path_to_data, f"{output_file}.json"), 'w') as file:
    #     json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
