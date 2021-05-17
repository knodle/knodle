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


def train_cleanlab(path_to_data: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10

    parameters = dict(
        # seed=None,
        lr=[0.1],
        cv_n_folds=[3, 5, 8],
        prune_method=['prune_by_class', 'prune_by_noise_rate', 'both'],
        epochs=[200],
        batch_size=[128],
        psx_calculation_method=['random', 'signatures', 'rules'],       # how the splitting into folds will be performed
    )
    parameter_values = [v for v in parameters.values()]

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    train_input_x, test_input_x, _ = get_tfidf_features(df_train["sample"], test_data=df_test["sample"])

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))

    test_labels = df_test["label"].tolist()
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1

    results = []
    for run_id, (lr, cv_n_folds, prune_method, epochs, batch_size, psx_calculation_method) in \
            enumerate(product(*parameter_values)):

        print("======================================")
        params = f'seed = None lr = {lr} cv_n_folds = {cv_n_folds} prune_method = {prune_method} epochs = {epochs} ' \
                 f'batch_size = {batch_size} psx_calculation_method = {psx_calculation_method}'
        print(f"Parameters: {params}")
        print("======================================")

        exp_results = []
        for exp in range(0, num_experiments):

            model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

            custom_cleanlab_config = CleanLabConfig(
                # seed=seed,
                cv_n_folds=cv_n_folds,
                output_classes=num_classes,
                optimizer=Adam,
                criterion=CrossEntropyLoss,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                psx_calculation_method=psx_calculation_method,
                noise_matrix="rule2class",
                prune_method=prune_method,
                use_probabilistic_labels=False
            )
            trainer = CleanLabTrainer(
                model=model,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=train_features_dataset,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config
            )

            trainer.train()
            clf_report, _ = trainer.test(test_features_dataset, test_labels_dataset)
            print(f"Accuracy is: {clf_report['accuracy']}")
            print(clf_report)

            exp_results.append(clf_report['accuracy'])

        results.append({
            # "seed": seed,
            "lr": lr, "cv_n_folds": cv_n_folds, "prune_method": prune_method, "epochs": epochs,
            "batch_size": batch_size, "psx_calculation_method": psx_calculation_method, "accuracy": exp_results,
            "mean_accuracy": statistics.mean(exp_results),
            "std_accuracy": statistics.stdev(exp_results)
        })

    with open(os.path.join(path_to_data, 'cl_results.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)
