import os

from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW

from knodle.model import LogisticRegressionModel
from joblib import load, dump
import pandas as pd
from torch import Tensor

from knodle.trainer.SimpleDsModelTrainer.SimpleDsModelTrainer import (
    SimpleDsModelTrainer,
)
from knodle.trainer.model_config.ModelConfig import ModelConfig


def train_simple_ds_model():
    print("Train simple ds model")
    imdb_dataset, applied_lfs = read_evaluation_data()
    tfidf_values = create_tfidf_values(imdb_dataset.reviews_preprocessed.values)

    tfidf_tensor = Tensor(tfidf_values.toarray())
    model = LogisticRegressionModel(tfidf_values.shape[1], 2)

    custom_model_config = ModelConfig(model=model, optimizer_=AdamW)

    trainer = SimpleDsModelTrainer(model, model_config=custom_model_config)
    trainer.train(inputs=tfidf_tensor, applied_labeling_functions=applied_lfs, epochs=2)


def read_evaluation_data():
    imdb_dataset = pd.read_csv("tutorials/ImdbDataset/imdb_data_preprocessed.csv")
    applied_lfs = load("tutorials/ImdbDataset/applied_lfs.lib")
    return imdb_dataset, applied_lfs


def create_tfidf_values(text_data: [str]):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib"):
        return load("tutorials/ImdbDataset/tfidf.lib")
    else:
        vectorizer = TfidfVectorizer()
        transformed_data = vectorizer.fit_transform(text_data)
        dump(transformed_data, "tutorials/ImdbDataset/tfidf.lib")
        return transformed_data


if __name__ == "__main__":
    train_simple_ds_model()
