import argparse
import json
import os
import statistics
import sys
from itertools import product

from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from examples.trainer.preprocessing import get_tfidf_features, convert_text_to_transformer_input
from examples.utils import read_train_dev_test
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def train_cleanlab_bert(path_to_data: str) -> None:
    """ This is an example of launching cleanlab trainer with BERT model """

    num_experiments = 10

    parameters = dict(
        # seed=None,
        lr=[0.1],
        cv_n_folds=[3, 5, 8],
        prune_method=['prune_by_noise_rate'],               # , 'prune_by_class', 'both'
        epochs=[2],
        batch_size=[32],
        psx_calculation_method=['random', 'signatures', 'rules'],      # how the splitting into folds will be performed
        psx_epochs=[20],
        psx_lr=[0.8]
    )
    parameter_values = [v for v in parameters.values()]

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    # create test labels dataset
    test_labels = df_test["label"].tolist()
    y_test = TensorDataset(LongTensor(test_labels))
    num_classes = max(test_labels) + 1

    # the psx matrix is calculated with logistic regression model (with TF-IDF features)
    train_input_x, test_input_x, _ = get_tfidf_features(df_train["sample"], test_data=df_test["sample"])
    X_train_tfidf = TensorDataset(Tensor(train_input_x.toarray()))

    # the classifier training is realized with BERT model (with BERT encoded features - input indices & attention mask)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_train_bert = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    X_test_bert = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    results = []
    for run_id, (params) in enumerate(product(*parameter_values)):

        lr, cv_n_folds, prune_method, epochs, batch_size, psx_calculation_method, psx_epochs, psx_lr = params

        print("======================================")
        params = f'seed = None lr = {lr} cv_n_folds = {cv_n_folds} prune_method = {prune_method} epochs = {epochs} ' \
                 f'batch_size = {batch_size} psx_calculation_method = {psx_calculation_method} ' \
                 f'psx_epochs = {psx_epochs} psx_lr = {psx_lr} '
        print(f"Parameters: {params}")
        print("======================================")

        exp_results_acc, exp_results_prec, exp_results_recall, exp_results_f1 = [], [], [], []

        for exp in range(0, num_experiments):

            model_logreg = LogisticRegressionModel(train_input_x.shape[1], num_classes)
            model_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                             num_labels=num_classes)

            custom_cleanlab_config = CleanLabConfig(
                cv_n_folds=cv_n_folds,
                output_classes=num_classes,
                optimizer=Adam,
                criterion=CrossEntropyLoss,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                psx_calculation_method=psx_calculation_method,
                prune_method=prune_method,
                use_probabilistic_labels=False,
                psx_epochs=psx_epochs,
                psx_lr=psx_lr
            )
            trainer = CleanLabTrainer(
                model=model_bert,
                mapping_rules_labels_t=mapping_rules_labels_t,
                model_input_x=X_train_bert,
                rule_matches_z=train_rule_matches_z,
                trainer_config=custom_cleanlab_config,

                psx_model=model_logreg,
                psx_model_input_x=X_train_tfidf
            )

            trainer.train()
            clf_report = trainer.test(X_test_bert, y_test)
            print(f"Accuracy is: {clf_report['accuracy']}")
            print(f"Precision is: {clf_report['macro avg']['precision']}")
            print(f"Recall is: {clf_report['macro avg']['recall']}")
            print(f"F1 is: {clf_report['macro avg']['f1-score']}")
            print(clf_report)

            exp_results_acc.append(clf_report['accuracy'])
            exp_results_prec.append(clf_report['macro avg']['precision'])
            exp_results_recall.append(clf_report['macro avg']['recall'])
            exp_results_f1.append(clf_report['macro avg']['f1-score'])

        result = {
            "lr": lr, "cv_n_folds": cv_n_folds, "prune_method": prune_method, "epochs": epochs,
            "batch_size": batch_size, "psx_calculation_method": psx_calculation_method,
            "accuracy": exp_results_acc,
            "mean_accuracy": statistics.mean(exp_results_acc), "std_accuracy": statistics.stdev(exp_results_acc),
            "precision": exp_results_prec,
            "mean_precision": statistics.mean(exp_results_prec), "std_precision": statistics.stdev(exp_results_prec),
            "recall": exp_results_recall,
            "mean_recall": statistics.mean(exp_results_recall), "std_recall": statistics.stdev(exp_results_recall),
            "f1-score": exp_results_f1,
            "mean_f1": statistics.mean(exp_results_f1), "std_f1": statistics.stdev(exp_results_f1),
        }
        results.append(result)

        print("======================================")
        print(f"Result: {result}")
        print("======================================")

    with open(os.path.join(path_to_data, 'cl_results_imdb.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab_bert(args.path_to_data)
