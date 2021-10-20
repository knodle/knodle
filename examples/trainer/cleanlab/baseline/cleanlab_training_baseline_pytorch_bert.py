import logging
import argparse
import os
import statistics
import sys
import json

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from scipy.stats import sem

from examples.trainer.preprocessing import convert_text_to_transformer_input
from examples.utils import read_train_dev_test
from knodle.trainer.cleanlab.cleanlab_base_with_pytorch import CleanlabBasePyTorchTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 30

    df_train, _, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=False)

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    # dev_labels = df_dev["label"].tolist()
    y_test = TensorDataset(LongTensor(test_labels))
    # y_dev = TensorDataset(LongTensor(dev_labels))

    num_classes = max(test_labels) + 1

    # the classifier training is realized with BERT model (with BERT encoded features - input indices & attention mask)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_train = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    # X_dev = convert_text_to_transformer_input(df_dev["sample"].tolist(), tokenizer)
    X_test = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], [], []

    for exp in range(0, num_experiments):

        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
        custom_cleanlab_config = CleanLabConfig(
            # seed=seed,
            cv_n_folds=5,
            use_prior=False,
            output_classes=num_classes,
            optimizer=AdamW,
            criterion=CrossEntropyLoss,
            use_probabilistic_labels=False,
            lr=0.0001,
            epochs=2,
            batch_size=32,
            grad_clipping=5,
            early_stopping=True
        )
        trainer = CleanlabBasePyTorchTrainer(
            model=model,
            mapping_rules_labels_t=mapping_rules_labels_t,
            model_input_x=X_train,
            rule_matches_z=train_rule_matches_z,
            trainer_config=custom_cleanlab_config,

            # dev_model_input_x=X_dev,
            # dev_gold_labels_y=y_dev
        )

        trainer.train()
        clf_report = trainer.test(X_test, y_test)

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
        f"Average F1: {result['mean_f1']}, std: {result['std_f1']}, sem: {result['sem_f1']}")
    print("======================================")

    with open(os.path.join(path_to_data, output_file), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
