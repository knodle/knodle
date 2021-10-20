import argparse
import os
import statistics
import sys
import json

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from scipy.stats import sem

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab_base_with_pytorch import CleanlabBasePyTorchTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 30

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=True)

    train_input_x, test_input_x, dev_input_x = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"],
    )

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    dev_features_dataset = TensorDataset(Tensor(dev_input_x.toarray()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))

    dev_labels = df_dev["label"].tolist()
    test_labels = df_test["label"].tolist()
    dev_labels_dataset = TensorDataset(LongTensor(dev_labels))
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1

    exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []
    for exp in range(0, num_experiments):

        model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

        custom_cleanlab_config = CleanLabConfig(
            # seed=seed,
            cv_n_folds=5,
            prune_method='prune_by_noise_rate',
            use_prior=False,
            output_classes=num_classes,
            optimizer=Adam,
            criterion=CrossEntropyLoss,
            use_probabilistic_labels=False,
            lr=0.1,
            epochs=100,
            batch_size=128,
            device="cpu",
            grad_clipping=5,
            early_stopping=True,
            save_model_name=output_file
        )
        trainer = CleanlabBasePyTorchTrainer(
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
        print(clf_report)

        exp_results_acc.append(clf_report['accuracy'])
        exp_results_prec.append(clf_report['macro avg']['precision'])
        exp_results_recall.append(clf_report['macro avg']['recall'])
        exp_results_f1.append(clf_report['macro avg']['f1-score'])

    result = {
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

    print("======================================")
    print(
        f"Experiments: {num_experiments} \n"
        f"Average accuracy: {result['mean_accuracy']}, std: {result['std_accuracy']}, sem: {result['sem_accuracy']} \n"
        f"Average prec: {result['mean_precision']}, std: {result['std_precision']}, sem: {result['sem_precision']} \n"
        f"Average recall: {result['mean_recall']}, std: {result['std_recall']}, sem: {result['sem_recall']} \n"
        f"Average F1: {result['std_f1']}, std: {result['std_f1']}, sem: {result['sem_f1']}")
    print("======================================")

    with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
