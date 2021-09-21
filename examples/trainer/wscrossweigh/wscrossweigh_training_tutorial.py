import argparse
import os
import sys

import joblib
import pandas as pd
from minio import Minio
from torch import Tensor, LongTensor
from torch.optim import Adam
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from examples.trainer.preprocessing import convert_text_to_transformer_input, get_tfidf_features
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.wscrossweigh.config import WSCrossWeighConfig
from knodle.trainer.wscrossweigh.wscrossweigh import WSCrossWeighTrainer


def train_wscrossweigh(path_to_data: str, num_classes: int) -> None:
    """
    We are going to train a BERT classification model using weakly annotated data with additional WSCrossWeigh
    denoising. The sample weights in WSCrossWeigh will be trained with logistic regression in order to, firstly,
    reduce the computational effort, and, secondly, demonstrate the ability of the algorithm to use different models
    for data denoising and classifier training.
    :param path_to_data: path to the folder where all the input data is stored
    :param num_classes: number of output classes
    """

    num_classes = int(num_classes)

    # Define constants
    imdb_data_dir = os.path.join(os.getcwd(), "datasets", "imdb")
    processed_data_dir = os.path.join(imdb_data_dir, "processed")
    os.makedirs(processed_data_dir, exist_ok=True)

    # Download data
    client = Minio("knodle.dm.univie.ac.at", secure=False)
    files = [
        "df_train.csv", "df_dev.csv", "df_test.csv",
        "train_rule_matches_z.lib", "dev_rule_matches_z.lib", "test_rule_matches_z.lib",
        "mapping_rules_labels_t.lib"
    ]
    for file in tqdm(files):
        client.fget_object(
            bucket_name="knodle",
            object_name=os.path.join("datasets/imdb/processed", file),
            file_path=os.path.join(processed_data_dir, file),
        )

    # Load data into memory
    df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"))
    df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"))
    df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"))

    mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))

    train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))

    # sample weights are calculated with logistic regression model (with TF-IDF features); the BERT model is used for
    # the final classifier training.
    train_tfidf_sparse, dev_tfidf_sparse, _ = get_tfidf_features(df_train["sample"].tolist(), df_dev["sample"].tolist())
    train_tfidf = Tensor(train_tfidf_sparse.toarray())
    train_dataset_tfidf = TensorDataset(train_tfidf)

    # For the BERT training we convert train, dev, test data to BERT encoded features (input indices & attention mask)
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    X_train = convert_text_to_transformer_input(df_train["sample"].tolist(), tokenizer)
    X_dev = convert_text_to_transformer_input(df_dev["sample"].tolist(), tokenizer)
    X_test = convert_text_to_transformer_input(df_test["sample"].tolist(), tokenizer)

    y_dev = TensorDataset(LongTensor(df_dev["label"].tolist()))
    y_test = TensorDataset(LongTensor(df_test["label"].tolist()))

    # define the all needed parameters in a dictionary for convenience (can also be directly passed to Trainer/Config)
    parameters = {
        "lr": 1e-4, "cw_lr": 0.8, "epochs": 5, "cw_partitions": 2, "cw_folds": 5, "cw_epochs": 2, "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }
    # to have sample weights saved with some specific index in the file name, you can use "caching_suffix" variable
    caching_suffix = f"dscw_{parameters.get('cw_partitions')}part_{parameters.get('cw_folds')}folds_" \
                     f"{parameters.get('weight_rr')}wrr"

    # define LogReg and BERT models for training sample weights and final classifier correspondingly
    cw_model = LogisticRegressionModel(train_tfidf.shape[1], num_classes)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    # define a custom WSCrossWeigh config. If no custom config is defined, the WSCrossWeighTrainer will use the default
    # WSCrossWeighConfig which is stored in the fold with the WSCrossWeigh trainer
    custom_wscrossweigh_config = WSCrossWeighConfig(
        # general trainer parameters
        output_classes=num_classes,
        filter_non_labelled=False,
        other_class_id=3,
        seed=12345,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=AdamW,
        lr=parameters.get("lr"),
        grad_clipping=5,
        caching_suffix=caching_suffix,
        saved_models_dir=os.path.join(path_to_data, "trained_models"),  # trained classifier model will be saved after each epoch

        # WSCrossWeigh specific parameters
        partitions=parameters.get("cw_partitions"),  # number of WSCrossWeigh iterations (= splitting into folds)
        folds=parameters.get("cw_folds"),  # number of folds train data will be splitted into
        weight_reducing_rate=parameters.get("weight_rr"),  # sample weights reducing coefficient
        samples_start_weights=parameters.get("samples_start_weights"),  # the start weight of sample weights
        cw_epochs=parameters.get("cw_epochs"),  # number of epochs each WSCrossWeigh model is to be trained
        cw_optimizer=Adam,  # WSCrossWeigh model optimiser
        cw_lr=parameters.get("cw_lr")  # WSCrossWeigh model lr
    )

    trainer = WSCrossWeighTrainer(
        # general Trainer inputs (a more detailed explanation of Knodle inputs is in README)
        model=model,  # classification model
        mapping_rules_labels_t=mapping_rules_labels_t,  # t matrix
        model_input_x=X_train,  # x matrix for training the classifier
        rule_matches_z=train_rule_matches_z,  # z matrix
        trainer_config=custom_wscrossweigh_config,

        # additional dev set used for classification model evaluation during training
        dev_model_input_x=X_dev,
        dev_gold_labels_y=y_dev,

        # WSCrossWeigh specific parameters. If they are not defined, the corresponding main classification parameters
        # will be used instead (model instead of cw_model etc)
        cw_model=cw_model,  # model that will be used for WSCrossWeigh weights calculation
        cw_model_input_x=train_dataset_tfidf,  # x matrix for training the WSCrossWeigh models
    )

    # the WSCrossWeighTrainer is trained
    trainer.train()
    # the trained model is tested on the test set
    clf_report, _ = trainer.test(X_test, y_test)
    print(clf_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="Path to the folder where all input files are stored.")
    parser.add_argument("--num_classes", help="Number of classes")
    args = parser.parse_args()

    train_wscrossweigh(args.path_to_data, args.num_classes)
