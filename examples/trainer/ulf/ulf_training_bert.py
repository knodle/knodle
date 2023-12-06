import logging
import argparse
import json
import os
import sys
from itertools import product

import pandas as pd
from joblib import load
from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, RobertaTokenizer, \
    RobertaForSequenceClassification

from examples.trainer.preprocessing import get_tfidf_features, convert_text_to_transformer_input
from examples.utils import read_train_dev_test, collect_res_statistics, print_metrics
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.ulf import UlfTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.utils import check_and_return_device

logger = logging.getLogger(__name__)


def train_ulf_bert(path_to_data: str, dataset: str, output_path: str, if_dev_data: bool = True) -> None:
    """ This is an example of launching cleanlab trainer with BERT model """

    device = check_and_return_device()
    save_model_path = os.path.join(output_path, f"trained_models_{dataset}")

    num_experiments = 10
    parameters = dict(
        # seed=None,
        use_prior=[False],
        psx_calculation_method=['signatures', 'rules', 'random'],      # how the splitting into folds will be performed
        p=[0.1, 0.3, 0.5, 0.7, 0.9],
        lr=[0.0001, 0.001, 0.01, 0.1],      # 0.00001,
        cv_n_folds=[3, 5, 8],           # , 10, 20
        iterations=[1, 2, 3, 5],            # 10
        other_coeff=[0, 0.5, 1, 3],         # 2
    )
    parameter_values = [v for v in parameters.values()]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    df_train = pd.read_csv(os.path.join(path_to_data, 'train_df.csv'), sep="\t")
    df_test = pd.read_csv(os.path.join(path_to_data, 'test_df.csv'), sep="\t")
    train_rule_matches_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    mapping_rules_labels_t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))

    # the psx matrix is calculated with logistic regression model (with TF-IDF features)
    train_input_x, test_input_x, _ = get_tfidf_features(df_train["sample"], test_data=df_test["sample"])

    if if_dev_data:
        df_dev = pd.read_csv(os.path.join(path_to_data, 'dev_df.csv'), sep="\t")
        # get dev BERT encodings
        X_dev_bert = convert_text_to_transformer_input(df_dev["sample"].tolist(), tokenizer)
        # create dev labels dataset
        dev_labels = df_dev["label"].tolist()
        dev_labels_dataset = TensorDataset(LongTensor(dev_labels))
    else:
        X_dev_bert, dev_labels_dataset = None, None

    # encode features for psx matrix calculation
    X_train_tfidf = TensorDataset(Tensor(train_input_x.toarray()))

    # the classifier training is realized with BERT model (with BERT encoded features - input indices & attention mask)
    X_train_bert = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    X_test_bert = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1
    results, exp_signatures = [], []

    for run_id, (params) in enumerate(product(*parameter_values)):
        use_prior, psx_method, p, lr, folds, iterations, other_coeff = params
        p = None if use_prior else p
        params_dict = {
            'prior': use_prior, 'epochs': 20, 'p': p, 'lr': lr, 'folds': folds, 'iter': iterations,
            'other_coeff': other_coeff, 'psx': psx_method
        }
        params_signature = str(params_dict)
        log_section(params_signature, logger)

        save_model_path_run = os.path.join(
            save_model_path, f"{use_prior}_{p}_{lr}_{folds}_{iterations}_{other_coeff}_{psx_method}"
        )

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted = \
            [], [], [], [], []

        for exp in range(0, num_experiments):
            model_logreg = LogisticRegressionModel(train_input_x.shape[1], num_classes)
            model_bert = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)

            custom_cleanlab_config = UlfConfig(
                iterations=iterations,
                cv_n_folds=folds,
                psx_calculation_method=psx_method,
                unmatched_strategy="random",
                use_probabilistic_labels=False,
                use_prior=use_prior,
                p=p,
                output_classes=num_classes,
                criterion=CrossEntropyLoss,
                epochs=20,
                grad_clipping=5,
                save_model_path=save_model_path_run,
                save_model_name=f"{dataset}_{exp}",
                optimizer=AdamW,
                lr=lr,
                batch_size=64,
                early_stopping=True,
                other_coeff=other_coeff,

                psx_epochs=20,
                psx_lr=0.8,
                psx_optimizer=Adam,

                device=device
            )
            logger.info(custom_cleanlab_config)
            trainer = UlfTrainer(
                model=model_bert,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=X_train_bert,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config,

                psx_model=model_logreg,
                psx_model_input_x=X_train_tfidf,

                dev_model_input_x=X_dev_bert,
                dev_gold_labels_y=dev_labels_dataset

                # df_train=df_train,
                # output_file=os.path.join(
                #     path_to_data, f"{output_file}_{exp}_{use_prior}_{p}_{folds}_{iterations}_{psx_method}"
                # )
            )

            trainer.train()
            clf_report = trainer.test(X_test_bert, test_labels_dataset)
            print_metrics(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1_avg.append(clf_report['macro avg']['f1-score'])
            exp_results_f1_weighted.append(clf_report['weighted avg']['f1-score'])

        exp_signatures.append(params_signature)

        res = collect_res_statistics(
                exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1_avg, exp_results_f1_weighted,
                params=params_dict, verbose=True
            )
        with open(os.path.join(output_path, f"{dataset}_final_results.json"), 'a+') as file:
            json.dump(res, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--dataset", help="")
    parser.add_argument("--output_path", help="")
    parser.add_argument("--if_dev_data", help="")

    args = parser.parse_args()

    if_dev_data = args.if_dev_data
    if if_dev_data == "True":
        if_dev_data_bool = True
    elif if_dev_data == "False":
        if_dev_data_bool = False
    else:
        raise ValueError("Invalid if_dev_data! Should be in boolean format.")

    train_ulf_bert(args.path_to_data, args.dataset, args.output_path, if_dev_data_bool)
