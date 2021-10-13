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
from knodle.trainer.cleanlab.cleanlab_with_pytorch import CleanLabPyTorchTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def train_cleanlab(path_to_data: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 30

    parameters = dict(
        # seed=None,
        lr=[0.1],
        cv_n_folds=[5],
        prune_method=['prune_by_noise_rate'],               # , 'prune_by_class', 'both'
        epochs=[100],
        batch_size=[128]
    )
    parameter_values = [v for v in parameters.values()]

    df_train, _, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=False)

    train_input_x, test_input_x, _ = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"]     #, dev_data=df_dev["sample"],
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
    for run_id, (lr, cv_n_folds, prune_method, epochs, batch_size) in enumerate(product(*parameter_values)):

        print("======================================")
        params = f'seed = None lr = {lr} cv_n_folds = {cv_n_folds} prune_method = {prune_method} epochs = {epochs} ' \
                 f'batch_size = {batch_size} '
        print(f"Parameters: {params}")
        print("======================================")

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []
        for exp in range(0, num_experiments):

            model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

            custom_cleanlab_config = CleanLabConfig(
                # seed=seed,
                cv_n_folds=cv_n_folds,
                prune_method=prune_method,
                use_prior=False,
                output_classes=num_classes,
                optimizer=Adam,
                criterion=CrossEntropyLoss,
                use_probabilistic_labels=False,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                device="cpu",
                grad_clipping=5
            )
            trainer = CleanLabPyTorchTrainer(
                model=model,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=train_features_dataset,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config,
                # dev_model_input_x=dev_features_dataset,
                # dev_gold_labels_y=dev_labels_dataset
            )

            trainer.train()
            clf_report = trainer.test(test_features_dataset, test_labels_dataset)
            print(f"Accuracy is: {clf_report['accuracy']}")
            print(f"Precision is: {clf_report['macro avg']['precision']}")
            print(f"Recall is: {clf_report['macro avg']['recall']}")
            print(f"F1 is: {clf_report['macro avg']['f1-score']}")
            print(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1.append(clf_report['macro avg']['f1-score'])

        result = {
            "lr": lr, "cv_n_folds": cv_n_folds, "prune_method": prune_method, "epochs": epochs,
            "batch_size": batch_size, "accuracy": exp_results_acc,
            "mean_accuracy": statistics.mean(exp_results_acc), "std_accuracy": statistics.stdev(exp_results_acc),
            "precision": exp_results_prec,
            "mean_precision": statistics.mean(exp_results_prec), "std_precision": statistics.stdev(exp_results_prec),
            "recall": exp_results_recall,
            "mean_recall": statistics.mean(exp_results_recall), "std_recall": statistics.stdev(exp_results_recall),
            "f1-score": exp_results_f1,
            "mean_f1": statistics.mean(exp_results_f1), "std_f1": statistics.stdev(exp_results_f1),
        }
        results.append(result)

        print("======================================")
        print(f"Result: {result}")
        print("======================================")

    with open(os.path.join(path_to_data, 'results/spouse/baselines/cl_results_spouse_baseline_pytorch_30exp.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)
