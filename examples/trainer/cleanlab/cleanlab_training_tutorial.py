import argparse
import os
import sys
from itertools import product

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset

from examples.trainer.dscrossweigh.dscrossweigh_training_tutorial import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def train_cleanlab(path_to_data: str) -> None:
    """ This is an example of launching cleanlab trainer """

    parameters = dict(
        seed=[12345], lr=[0.1], cv_n_folds=[3, 5, 8], prune_method=['prune_by_class', 'prune_by_noise_rate', 'both'],
        epochs=[200], batch_size=[128], psx_calculation_method=['split_by_signatures', 'split_by_rules', 'random'],
    )
    parameter_values = [v for v in parameters.values()]

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    train_input_x, test_input_x, _ = get_tfidf_features(df_train["sample"], test_data=df_test["sample"])

    train_features_dataset = TensorDataset(Tensor(train_input_x.toarray()))
    test_features_dataset = TensorDataset(Tensor(test_input_x.toarray()))

    test_labels = list(df_test.iloc[:, -1])
    test_labels_dataset = TensorDataset(LongTensor(list(df_test.iloc[:, 1])))

    num_classes = max(test_labels) + 1
    model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

    for run_id, (seed, lr, cv_n_folds, prune_method, epochs, batch_size, psx_calculation_method) in \
            enumerate(product(*parameter_values)):

        print("======================================")
        params = f'seed = {seed} lr = {lr} cv_n_folds = {cv_n_folds} prune_method = {prune_method} epochs = {epochs} ' \
                 f'batch_size = {batch_size} psx_calculation_method = {psx_calculation_method}'
        print(f"Parameters: {params}")
        print("======================================")

        custom_cleanlab_config = CleanLabConfig(
            seed=seed,
            cv_n_folds=cv_n_folds,
            output_classes=num_classes,
            optimizer=Adam,
            criterion=CrossEntropyLoss,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            psx_calculation_method=psx_calculation_method,
            prune_method=prune_method
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
        print(clf_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)
