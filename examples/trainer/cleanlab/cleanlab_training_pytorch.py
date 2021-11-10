import logging
import argparse
import json
import os
import sys

import pandas as pd
from joblib import load
from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from torch.optim import Adam

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test, collect_res_statistics, print_metrics
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanlabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10

    train_rule_matches_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))

    df_train = pd.read_csv(os.path.join(path_to_data, "df_train.csv"))
    df_test = pd.read_csv(os.path.join(path_to_data, "df_test.csv"))
    df_dev = pd.read_csv(os.path.join(path_to_data, "df_dev.csv"))

    train_input_x, test_input_x, dev_input_x = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"]
    )

    # df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
    #     path_to_data, if_dev_data=True)
    #
    # train_input_x, test_input_x, dev_input_x = get_tfidf_features(
    #     df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"],
    # )

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    dev_features_dataset = TensorDataset(Tensor(dev_input_x.toarray()))
    dev_labels_dataset = TensorDataset(LongTensor(df_dev["label"].tolist()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))
    test_labels_dataset = TensorDataset(LongTensor(df_test["label"].tolist()))

    num_classes = max(df_test["label"].tolist()) + 1

    results = []
    for curr_psx_method in ["random"]:
        for curr_lr in [0.01]:
            params_dict = {'psx': curr_psx_method, 'lr': curr_lr}
            log_section(str(params_dict), logger)

            exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted = \
                [], [], [], [], []

            for exp in range(0, num_experiments):
                logger.info(f"Experiment {exp} out of {num_experiments}")
                model = LogisticRegressionModel(train_input_x.shape[1], num_classes)
                custom_cleanlab_config = CleanLabConfig(
                    cv_n_folds=5,
                    psx_calculation_method=curr_psx_method,
                    output_classes=num_classes,
                    filter_non_labelled=False,
                    optimizer=Adam,
                    criterion=CrossEntropyLoss,
                    lr=curr_lr,
                    epochs=30,
                    batch_size=128,
                    grad_clipping=5,
                    early_stopping=True,
                    save_model_name=f"{output_file}_{curr_psx_method}_{curr_lr}_{str(exp)}"
                )
                logger.info(custom_cleanlab_config)
                trainer = CleanlabTrainer(
                    model=model,
                    mapping_rules_labels_t=mapping_rules_labels_t,
                    model_input_x=train_features_dataset,
                    rule_matches_z=train_rule_matches_z,
                    trainer_config=custom_cleanlab_config,
                    dev_model_input_x=dev_features_dataset,
                    dev_gold_labels_y=dev_labels_dataset
                )
                trainer.train()
                clf_report = trainer.test(test_features_dataset, test_labels_dataset)
                print_metrics(clf_report)

                exp_results_acc.append(clf_report['accuracy'])
                exp_results_prec.append(clf_report['macro avg']['precision'])
                exp_results_recall.append(clf_report['macro avg']['recall'])
                exp_results_f1_avg.append(clf_report['macro avg']['f1-score'])
                exp_results_f1_weighted.append(clf_report['weighted avg']['f1-score'])

            results.append(
                collect_res_statistics(
                    exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted,
                    params=params_dict, verbose=True
                )
            )

    with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
