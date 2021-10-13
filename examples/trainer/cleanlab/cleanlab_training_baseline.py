import argparse
import json
import os
import statistics
import sys
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import dataset_to_numpy_input


def train_cleanlab(path_to_data: str) -> None:
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

    exp_results_acc, exp_results_f1 = [], []
    for exp in range(0, num_experiments):
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
        rp = LearningWithNoisyLabels(clf=clf, seed=seed, n_jobs=1)
        _ = rp.fit(train_input_x, train_labels)
        pred = rp.predict(test_input_x)

        accuracy = accuracy_score(pred, test_labels)
        f1 = f1_score(pred, test_labels)

        print(f"Accuracy is: {accuracy}")
        print(f"F1 is: {f1}")

        exp_results_acc.append(accuracy)
        exp_results_f1.append(f1)

    result = {
        "accuracy": exp_results_acc, "mean_accuracy": statistics.mean(exp_results_acc),
        "std_accuracy": statistics.stdev(exp_results_acc),
        "f1-score": exp_results_f1, "mean_f1": statistics.mean(exp_results_f1),
        "std_f1": statistics.stdev(exp_results_f1),
    }

    print("======================================")
    print(f"Average accuracy after {num_experiments} experiments: {statistics.mean(exp_results_acc)}, "
          f"std: {statistics.stdev(exp_results_acc)}"
          f"Average F1 after {num_experiments} experiments: {statistics.mean(exp_results_f1)}"
          f"std: {statistics.stdev(exp_results_f1)}")
    print("======================================")

    with open(os.path.join(path_to_data, 'cl_results_spam_sklearn_baseline.json'), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)