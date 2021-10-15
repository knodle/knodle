import logging
import argparse
import os
import statistics
import sys
import json
from itertools import product

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 30

    parameters = dict(
        # seed=None,
        cv_n_folds=[5],
        p=[0.1],
        use_prior=[False],
        iterations=[50],
        prune_method=['prune_by_noise_rate'],               # , 'prune_by_class', 'both'
        epochs=[100],
        psx_calculation_method=['signatures'],      # how the splitting into folds will be performed
    )
    parameter_values = [v for v in parameters.values()]

    # for trec dataset
    import pandas as pd
    from joblib import load
    df_train = pd.read_csv(os.path.join(path_to_data, 'df_train.csv'))
    df_test = pd.read_csv(os.path.join(path_to_data, 'df_test.csv'))
    train_rule_matches_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))
    df_dev = pd.read_csv(os.path.join(path_to_data, 'df_dev.csv'))

    # df_train, _, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
    #     path_to_data, if_dev_data=False)

    train_input_x, test_input_x, _ = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"]         #, dev_data=df_dev["sample"]
    )

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    # dev_features_dataset = TensorDataset(Tensor(dev_input_x.toarray()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))

    # dev_labels = df_dev["label"].tolist()
    test_labels = df_test["label"].tolist()
    # dev_labels_dataset = TensorDataset(LongTensor(dev_labels))
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1

    results = []
    for run_id, (cv_n_folds, p, use_prior, iterations, prune_method, epochs, psx_calculation_method) in\
            enumerate(product(*parameter_values)):

        logger.info("======================================")
        logger.info(f"Parameters: seed = None cv_n_folds = {cv_n_folds} iterations = {iterations} "
                    f"prune_method = {prune_method} epochs = {epochs} psx_calculation_method = {psx_calculation_method}"
                    f"p = {p} use_prior = {use_prior}")
        logger.info("======================================")

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []
        for exp in range(0, num_experiments):

            model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

            custom_cleanlab_config = CleanLabConfig(
                # seed=seed,
                cv_n_folds=cv_n_folds,
                psx_calculation_method=psx_calculation_method,
                prune_method=prune_method,
                iterations=iterations,
                use_prior=use_prior,
                p=p,
                output_classes=num_classes,
                optimizer=Adam,
                criterion=CrossEntropyLoss,
                use_probabilistic_labels=False,
                lr=0.1,
                epochs=epochs,
                batch_size=128,
                grad_clipping=5,
                save_model_name="trec",
                early_stopping=True
                # device="cpu"
            )
            trainer = CleanLabTrainer(
                model=model,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=train_features_dataset,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config,

                dev_model_input_x=test_features_dataset,
                dev_gold_labels_y=test_labels_dataset
            )

            trainer.train()
            clf_report = trainer.test(test_features_dataset, test_labels_dataset)       # , load_best_model=True)
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
            "cv_n_folds": cv_n_folds, "p": p, "prune_method": prune_method, "epochs": epochs,
            "use_prior": use_prior, "psx_calculation_method": psx_calculation_method,
            "accuracy": exp_results_acc,
            "mean_accuracy": statistics.mean(exp_results_acc), "std_accuracy": statistics.stdev(exp_results_acc),
            "precision": exp_results_prec,
            "mean_precision": statistics.mean(exp_results_prec), "std_precision": statistics.stdev(exp_results_prec),
            "recall": exp_results_recall,
            "mean_recall": statistics.mean(exp_results_recall), "std_recall": statistics.stdev(exp_results_recall),
            "f1-score": exp_results_f1,
            "mean_f1": statistics.mean(exp_results_f1), "std_f1": statistics.stdev(exp_results_f1),
        }
        results.append(result)

        logger.info("======================================")
        logger.info(f"Result: {result}")
        logger.info("======================================")

    with open(os.path.join(path_to_data, '___.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)
