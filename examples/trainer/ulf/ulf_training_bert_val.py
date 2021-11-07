import logging
import argparse
import json
import os
import sys
from itertools import product

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

from examples.trainer.preprocessing import get_tfidf_features, convert_text_to_transformer_input
from examples.utils import read_train_dev_test, collect_res_statistics, print_metrics
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.ulf import UlfTrainer
from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)


def train_cleanlab_bert(path_to_data: str, output_file: str) -> None:
    """ This is an example of launching cleanlab trainer with BERT model """

    num_experiments = 30

    parameters = dict(
        # seed=None,
        cv_n_folds=[3, 5, 8],
        p=[0.1, 0.3, 0.5, 0.7, 0.9],
        use_prior=[False],
        iterations=[50],
        psx_calculation_method=['signatures'],      # how the splitting into folds will be performed
    )
    parameter_values = [v for v in parameters.values()]

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(
        path_to_data, if_dev_data=True)

    # the psx matrix is calculated with logistic regression model (with TF-IDF features)
    train_input_x, test_input_x, dev_input_x = get_tfidf_features(
        df_train["sample"], test_data=df_test["sample"], dev_data=df_dev["sample"],
    )
    X_train_tfidf = TensorDataset(Tensor(train_input_x.toarray()))

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    dev_labels = df_dev["label"].tolist()
    dev_labels_dataset = TensorDataset(LongTensor(dev_labels))
    test_labels_dataset = TensorDataset(LongTensor(test_labels))

    num_classes = max(test_labels) + 1

    # the classifier training is realized with BERT model (with BERT encoded features - input indices & attention mask)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_train_bert = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    X_dev_bert = convert_text_to_transformer_input(df_dev["sample"].tolist(), tokenizer)
    X_test_bert = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    results = []
    for run_id, (params) in enumerate(product(*parameter_values)):

        cv_n_folds, p, use_prior, iterations, psx_calculation_method = params

        log_section(f"folds = {cv_n_folds} psx_method = {psx_calculation_method} p = {p} prior = {use_prior}", logger)

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []

        for exp in range(0, num_experiments):

            model_logreg = LogisticRegressionModel(train_input_x.shape[1], num_classes)

            model_bert = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', num_labels=num_classes
            )

            custom_cleanlab_config = UlfConfig(
                cv_n_folds=cv_n_folds,
                psx_calculation_method=psx_calculation_method,
                iterations=iterations,
                use_prior=use_prior,
                p=p,
                output_classes=num_classes,
                criterion=CrossEntropyLoss,
                use_probabilistic_labels=False,
                epochs=2,
                grad_clipping=5,
                save_model_name=output_file,
                optimizer=AdamW,
                lr=0.0001,
                batch_size=32,
                early_stopping=True,

                psx_epochs=20,
                psx_lr=0.8,
                psx_optimizer=Adam
            )

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
            )

            trainer.train()
            clf_report = trainer.test(X_test_bert, test_labels_dataset)
            print_metrics(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1.append(clf_report['macro avg']['f1-score'])

        results.append(
            collect_res_statistics(exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1, verbose=True)
        )

        with open(os.path.join(path_to_data, output_file + ".json"), 'w') as file:
            json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_file", help="")

    args = parser.parse_args()
    train_cleanlab_bert(args.path_to_data, args.output_file)
