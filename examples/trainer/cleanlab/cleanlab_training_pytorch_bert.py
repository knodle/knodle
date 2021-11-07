import logging
import argparse
import json
import os
import sys

from torch import LongTensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

from examples.trainer.preprocessing import convert_text_to_transformer_input
from examples.utils import read_train_dev_test, collect_res_statistics, print_metrics
from knodle.trainer.cleanlab.cleanlab import CleanlabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)


def train_cleanlab(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer """

    num_experiments = 10

    df_train, _, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=False)

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    # dev_labels = df_dev["label"].tolist()
    test_labels_dataset = TensorDataset(LongTensor(test_labels))
    # dev_labels_dataset = TensorDataset(LongTensor(dev_labels))

    # the classifier training is realized with BERT model (with BERT encoded features - input indices & attention mask)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_features_dataset = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    # X_dev = convert_text_to_transformer_input(df_dev["sample"].tolist(), tokenizer)
    test_features_dataset = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    num_classes = max(df_test["label"].tolist()) + 1

    results = []
    for curr_psx_method in ["random", "signatures", "rules"]:
        params_dict = {'psx': curr_psx_method}
        log_section(str(params_dict), logger)

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []

        for exp in range(0, num_experiments):
            logger.info(f"Experiment {exp} out of {num_experiments}")
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
            custom_cleanlab_config = CleanLabConfig(
                cv_n_folds=5,
                psx_calculation_method=curr_psx_method,
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
            trainer = CleanlabTrainer(
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
            print_metrics(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1.append(clf_report['macro avg']['f1-score'])

        results.append(
            collect_res_statistics(
                exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1, params=params_dict, verbose=True
            )
        )

    with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data, args.output_file)
