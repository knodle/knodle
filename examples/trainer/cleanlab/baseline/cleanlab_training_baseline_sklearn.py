import argparse
import json
import os
import statistics
import sys
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np
from scipy.stats import sem

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import dataset_to_numpy_input


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10
    seed = None

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)
    train_input_x, test_input_x, _ = get_tfidf_features(df_train["sample"], test_data=df_test["sample"])

    train_input_x, test_input_x = train_input_x.toarray(), test_input_x.toarray()
    train_labels = z_t_matrices_to_majority_vote_probs(train_rule_matches_z, mapping_rules_labels_t)
    train_input_x, train_labels, _ = filter_empty_probabilities(
        TensorDataset(Tensor(train_input_x)), train_labels, train_rule_matches_z
    )
    train_input_x = dataset_to_numpy_input(train_input_x)

    train_labels = np.argmax(train_labels, axis=1)
    test_labels = df_test["label"].to_numpy()

    exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []

    for exp in range(0, num_experiments):

        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
        rp = LearningWithNoisyLabels(clf=clf, seed=seed, n_jobs=1)
        _ = rp.fit(train_input_x, train_labels)
        pred = rp.predict(test_input_x)

        clf_report = classification_report(y_true=test_labels, y_pred=pred, output_dict=True)
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

    with open(os.path.join(path_to_data, output_file), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
