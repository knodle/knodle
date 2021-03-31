import os
from joblib import load, dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW

from knodle.model.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer import TrainerConfig
from knodle.trainer.knn_denoising.knn import (
    KnnDenoisingTrainer,
)
from torch import Tensor
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from minio import Minio
from dotenv import load_dotenv

load_dotenv()


def get_minio_config():
    config = {
        "minio_url": "knodle.dm.univie.ac.at",
        "minio_user": "UnM_LN*jSYK74Iz4",
        "minio_pw": "cQOs4|9Dr2_+HuFKneC8@dRgAtrV21i4Dumy",
        "minio_bucket": "knodle",
    }
    return config


def get_imdb_config():
    config = {
        "minio_prefix": "datasets/imdb",
        "minio_files": [
            "imdb_data_preprocessed.csv",
            "mapping_rules_labels.lib",
            "rule_matches.lib",
        ],
        "data_dir": "imdb_experiments_data",
        "num_features": 400,
        "num_classes": 2,
    }
    return config


def get_conll_config():
    config = {
        "minio_prefix": "datasets/conll/preprocessed",
        "minio_files": [
            "z_matrix.lib",
            "t_matrix.lib",
            "train_samples.csv",
            "dev_samples.csv",
        ],
        "data_dir": "conll_experiments_data",
        "num_features": 400,
        "num_classes": 2,
    }
    return config


def get_config(data_source="imdb"):
    if data_source == "imdb":
        config = get_imdb_config()
    elif data_source == "conll":
        config = get_conll_config()
    else:
        raise ValueError(
            "Please provide a valid data source. Currently supported are ['imdb', 'conll']"
        )

    config.update(get_minio_config())
    return config


def config_env(data_source: str = "imdb"):
    config = get_config(data_source)
    os.environ["data_source"] = data_source

    data_dir = os.path.join(os.getcwd(), config.get("data_dir"))
    os.environ["data_dir"] = data_dir
    os.makedirs(os.path.join(data_dir, "imdb_data"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "imdb_results"), exist_ok=True)


def download_conll():
    config = get_config("conll")

    # check if data is already cached
    if all(
            [
                os.path.isfile(os.path.join(os.getenv("data_dir"), file))
                for file in config.get("minio_files")
            ]
    ):
        return

    client = Minio(config.get("minio_url"), secure=False)

    for file in tqdm(config.get("minio_files")):
        client.fget_object(
            bucket_name=config.get("minio_bucket"),
            object_name=os.path.join(config.get("minio_prefix"), file),
            file_path=os.path.join(os.getenv("data_dir"), "conll_data", file),
        )


def load_conll_data():
    # data location
    data_dir = os.getenv("data_dir", False)
    if not data_dir:
        raise ValueError("Provide a data directory.")

    # load into memory
    conll_dataset_train = pd.read_csv(
        os.path.join(data_dir, "conll_data", "train_samples.csv")
    )
    conll_dataset_dev = pd.read_csv(
        os.path.join(data_dir, "conll_data", "dev_samples.csv")
    )
    rule_matches_z = load(os.path.join(data_dir, "conll_data", "z_matrix.lib"))
    mapping_rules_labels_t = load(os.path.join(data_dir, "conll_data", "t_matrix.lib"))

    return (
        conll_dataset_train,
        conll_dataset_dev,
        rule_matches_z,
        mapping_rules_labels_t,
    )


def create_tfidf_values(text_data: [str]):
    data_dir = os.getenv("data_dir", False)
    if not data_dir:
        raise ValueError("Provide a data directory.")

    if os.path.exists(f"{data_dir}/conll_data/tfidf_cached.lib"):
        cached_data = load(f"{data_dir}/conll_data/tfidf_cached.lib")
        if cached_data.shape == text_data.shape:
            return cached_data

    config = get_imdb_config()
    vectorizer = TfidfVectorizer(max_features=config.get("num_features"))
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, f"{data_dir}/conll_data/tfidf_cached.lib")
    return transformed_data


def preprocess_conll(train, dev, rule_matches_z, mapping_rules_labels_t):
    train_texts = train.samples.values
    dev_texts = dev.samples.values

    # preprocess and split data
    train_tfidf_values = create_tfidf_values(train_texts)
    dev_tfidf_values = create_tfidf_values(dev_texts)

    # transform data type
    train_tfidf = TensorDataset(Tensor(train_tfidf_values.toarray()))
    train_rule_matches_z = rule_matches_z[train.index]
    y_train = TensorDataset(Tensor(train.enc_labels.values))

    test_tfidf = TensorDataset(Tensor(dev_tfidf_values.toarray()))
    y_test = TensorDataset(Tensor(dev.enc_labels.values))

    return (
        train_tfidf,
        train_rule_matches_z,
        mapping_rules_labels_t,
        y_train,
        test_tfidf,
        y_test,
        train_tfidf_values,
    )


def train(train_tfidf, train_rule_matches_z, mapping_rules_labels_t, tfidf_values):
    config = get_conll_config()
    model = LogisticRegressionModel(
        config.get("num_features"), config.get("num_classes")
    )

    custom_model_config = TrainerConfig(model=model, epochs=25)

    trainer = KnnDenoisingTrainer(
        model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=train_tfidf,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_model_config,
        k=20,
        tfidf_values=tfidf_values,
    )
    trainer.train()

    return trainer


def test(trainer, train_tfidf, test_tfidf, y_train, y_test):
    results_dict_train_split, _ = trainer.test(train_tfidf, y_train)
    results_dict_test_split, _ = trainer.test(test_tfidf, y_test)

    return results_dict_train_split, results_dict_test_split


def run_conll():
    # download data
    download_conll()

    # load and preprocess data
    (
        conll_dataset_train,
        conll_dataset_dev,
        rule_matches_z,
        mapping_rules_labels_t,
    ) = load_conll_data()
    (
        train_tfidf,
        train_rule_matches_z,
        mapping_rules_labels_t,
        y_train,
        test_tfidf,
        y_test,
        tfidf_values_sparse,
    ) = preprocess_conll(
        conll_dataset_train, conll_dataset_dev, rule_matches_z, mapping_rules_labels_t
    )

    # Train
    trainer = train(
        train_tfidf, train_rule_matches_z, mapping_rules_labels_t, tfidf_values_sparse
    )

    # Test
    results_dict_train_split, results_dict_test_split = test(
        trainer, train_tfidf, test_tfidf, y_train, y_test
    )

    return results_dict_train_split, results_dict_test_split


if __name__ == "__main__":
    run_conll()
