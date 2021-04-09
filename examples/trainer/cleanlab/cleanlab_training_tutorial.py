import argparse
import os
import sys

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset

from examples.trainer.crossweigh_weighing_example.dscrossweigh_training_tutorial import get_tfidf_features
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def train_cleanlab(path_to_data: str, num_classes: int) -> None:
    """ This is an example of launching cleanlab trainer """

    parameters = {"seed": 12345, "lr": 0.1, "epochs": 10, "cv_n_folds": 3, "batch_size": 128}

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    train_input_x, test_input_x, _ = get_tfidf_features(df_train, column_num=0, test_data=df_test)
    train_dataset = TensorDataset(Tensor(train_input_x.toarray()))

    test_dataset = TensorDataset(Tensor(test_input_x.toarray()))
    test_labels = TensorDataset(LongTensor(list(df_test.iloc[:, 1])))

    model = LogisticRegressionModel(train_input_x.shape[1], num_classes)

    custom_cleanlab_config = CleanLabConfig(
        seed=parameters.get("seed"),
        cv_n_folds=parameters.get("cv_n_folds"),
        output_classes=num_classes,
        optimizer=Adam,
        criterion=CrossEntropyLoss,
        lr=parameters.get("lr"),
        epochs=parameters.get("epochs"),
        batch_size=parameters.get("batch_size")
    )

    trainer = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=train_dataset,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_cleanlab_config,
        test_x=test_dataset,
        test_y=test_labels
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)
    print(clf_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--num_classes", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.num_classes)
