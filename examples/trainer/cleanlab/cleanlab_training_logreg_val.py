import logging
import argparse
import json
import os
import statistics
import sys
from itertools import product

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from scipy.stats import sem

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 50

    parameters = dict(
        # seed=None,
        cv_n_folds=[3],     #, 5, 8],
        p=[0.9],            # 0.3, 0.5, 0.7, 0.9],
        use_prior=[False],
        iterations=[50],
        psx_calculation_method=['signatures'],      # how the splitting into folds will be performed
    )
    parameter_values = [v for v in parameters.values()]

    # for trec dataset
    # import pandas as pd
    # from joblib import load
    # df_train = pd.read_csv(os.path.join(path_to_data, 'df_train.csv'))
    # df_test = pd.read_csv(os.path.join(path_to_data, 'df_test.csv'))
    # train_rule_matches_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    # mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))
    # df_dev = pd.read_csv(os.path.join(path_to_data, 'df_dev.csv'))

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=True)

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

    results = []
    for run_id, (params) in enumerate(product(*parameter_values)):

        cv_n_folds, p, use_prior, iterations, psx_calculation_method = params

        logger.info("======================================")
        logger.info(f"Parameters: cv_n_folds = {cv_n_folds} psx_calculation_method = {psx_calculation_method}"
                    f"p = {p} use_prior = {use_prior}")
        logger.info("======================================")

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []

        for exp in range(0, num_experiments):

            model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

            custom_cleanlab_config = CleanLabConfig(
                # seed=seed,
                cv_n_folds=cv_n_folds,
                psx_calculation_method=psx_calculation_method,
                iterations=iterations,
                use_prior=use_prior,
                p=p,
                output_classes=num_classes,
                criterion=CrossEntropyLoss,
                use_probabilistic_labels=False,
                epochs=100,
                grad_clipping=5,
                save_model_name=output_file,
                optimizer=Adam,
                lr=0.1,
                batch_size=128,
                early_stopping=True
            )
            trainer = CleanLabTrainer(
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
            logger.info(f"Accuracy is: {clf_report['accuracy']}")
            logger.info(f"Precision is: {clf_report['macro avg']['precision']}")
            logger.info(f"Recall is: {clf_report['macro avg']['recall']}")
            logger.info(f"F1 is: {clf_report['macro avg']['f1-score']}")
            logger.info(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1.append(clf_report['macro avg']['f1-score'])

        result = {
            "cv_n_folds": cv_n_folds, "p": p, "psx_calculation_method": psx_calculation_method,
            "accuracy": exp_results_acc,
            "mean_accuracy": statistics.mean(exp_results_acc), "std_accuracy": statistics.stdev(exp_results_acc),
            "sem_accuracy": sem(exp_results_acc),
            "precision": exp_results_prec,
            "mean_precision": statistics.mean(exp_results_prec), "std_precision": statistics.stdev(exp_results_prec),
            "sem_precision": sem(exp_results_prec),
            "recall": exp_results_recall,
            "mean_recall": statistics.mean(exp_results_recall), "std_recall": statistics.stdev(exp_results_recall),
            "sem_recall": sem(exp_results_recall),
            "f1-score": exp_results_f1,
            "mean_f1": statistics.mean(exp_results_f1), "std_f1": statistics.stdev(exp_results_f1),
            "sem_f1": sem(exp_results_f1),
        }
        results.append(result)

        logger.info("======================================")
        logger.info(f"Params: cv_n_folds = {result['cv_n_folds']}, prior = {custom_cleanlab_config.use_prior}, "
                    f"p = {result['p']}")
        logger.info(
            f"Experiments: {num_experiments} \n"
            f"Average accuracy: {result['mean_accuracy']}, std: {result['std_accuracy']}, "
            f"sem: {result['sem_accuracy']} \n"
            f"Average prec: {result['mean_precision']}, std: {result['std_precision']}, "
            f"sem: {result['sem_precision']} \n"
            f"Average recall: {result['mean_recall']}, std: {result['std_recall']}, "
            f"sem: {result['sem_recall']} \n"
            f"Average F1: {result['std_f1']}, std: {result['std_f1']}, "
            f"sem: {result['sem_f1']}")
        logger.info("======================================")

    with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
