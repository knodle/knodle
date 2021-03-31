import argparse
import os
import sys
from typing import Union

from skorch import NeuralNetClassifier
from cleanlab.classification import LearningWithNoisyLabels
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from examples.trainer.crossweigh_weighing_example.dscrossweigh_training_tutorial import get_tfidf_features
from examples.utils import read_train_dev_test, create_bert_encoded_features, create_bert_encoded_features_hidden_states
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs


def train_cleanlab(path_to_data: str):

    seed = 12345
    num_classes = 2
    lr = 1e-4
    epochs = 1

    df_train, df_dev, df_test, train_rule_matches_z, _, mapping_rules_labels_t = read_train_dev_test(path_to_data)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_classes, output_hidden_states=True
    )

    y_test = list(df_test.iloc[:, 1])
    X_test = create_bert_encoded_features_hidden_states(df_test, tokenizer, 0)

    X_train = create_bert_encoded_features_hidden_states(df_train, tokenizer, 0)
    noisy_y_train = np.argmax(z_t_matrices_to_majority_vote_probs(train_rule_matches_z, mapping_rules_labels_t), axis=1)

    cls = NeuralNetClassifier(model, max_epochs=epochs, lr=lr)    # sklearn wrapper

    rp = LearningWithNoisyLabels(clf=cls, seed=seed)
    _ = rp.fit(X_train, noisy_y_train)
    pred = rp.predict(X_test)
    print("Test accuracy:", round(accuracy_score(pred, y_test), 2))


def get_tfidf_data(
        df_train: Union[pd.Series, pd.DataFrame], df_test: Union[pd.Series, pd.DataFrame], rule_matches_z: np.ndarray,
        mapping_rules_labels_t: np.ndarray
):
    X_train, X_test, _ = get_tfidf_features(df_train, df_test, 0)
    y_test = list(df_test.iloc[:, 1])
    noisy_y_train = np.argmax(z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t), axis=1)

    return X_train, noisy_y_train, X_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")

    args = parser.parse_args()
    train_cleanlab(args.path_to_data)
