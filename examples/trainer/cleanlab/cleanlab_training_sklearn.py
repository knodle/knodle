import argparse
import json
import os
import sys

import pandas as pd
from joblib import load
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from examples.trainer.preprocessing import get_tfidf_features
from examples.utils import collect_res_statistics, print_metrics
from knodle.transformation.majority import input_to_majority_vote_input


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10
    seed = None

    train_rule_matches_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    # z_test_rule_matches = load(os.path.join(path_to_data, 'test_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))

    df_train = pd.read_csv(os.path.join(path_to_data, "df_train.csv"))
    df_test = pd.read_csv(os.path.join(path_to_data, "df_test.csv"))
    df_dev = pd.read_csv(os.path.join(path_to_data, "df_dev.csv"))

    train_input_x, test_input_x, dev_input_x = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"]
    )
    train_input_x, dev_input_x, test_input_x = train_input_x.toarray(), dev_input_x.toarray(), test_input_x.toarray()

    _, train_labels, _ = input_to_majority_vote_input(
        train_rule_matches_z, mapping_rules_labels_t, use_probabilistic_labels=False
    )

    dev_labels = df_dev["label"].to_numpy()
    test_labels = df_test["label"].to_numpy()

    exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted = \
        [], [], [], [], []

    for exp in range(0, num_experiments):

        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
        rp = LearningWithNoisyLabels(clf=clf, seed=seed, n_jobs=1)
        _ = rp.fit(train_input_x, train_labels)
        pred = rp.predict(test_input_x)

        clf_report = classification_report(y_true=test_labels, y_pred=pred, output_dict=True)
        print_metrics(clf_report)

        exp_results_acc.append(clf_report['accuracy'])
        exp_results_prec.append(clf_report['macro avg']['precision'])
        exp_results_recall.append(clf_report['macro avg']['recall'])
        exp_results_f1_avg.append(clf_report['macro avg']['f1-score'])
        exp_results_f1_weighted.append(clf_report['weighted avg']['f1-score'])

    result = collect_res_statistics(
        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted,
        verbose=True
    )

    with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
