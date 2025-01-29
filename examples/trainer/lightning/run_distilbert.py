import os

from minio import Minio
from tqdm.auto import tqdm

import torchmetrics

from knodle.trainer.lightning.data_module import KnodleDataModule
from knodle.trainer.lightning.base_model import KnodleLightningModule
from knodle.trainer.lightning.majority_trainer import LightningMajorityTrainer
from knodle.trainer.lightning.snorkel_trainer import LightningSnorkelTrainer

from utils import get_tfidf_features, get_transformer_features


def main(dataset: str):

    imdb_data_dir = os.path.join(os.getcwd(), "datasets", dataset)
    processed_data_dir = os.path.join(imdb_data_dir, "processed")
    os.makedirs(processed_data_dir, exist_ok=True)

    client = Minio("knodle.cc", secure=False)
    files = [
        "df_train.csv", "df_test.csv",
        "train_rule_matches_z.pbz2", "test_rule_matches_z.pbz2",
        "mapping_rules_labels_t.lib",
        "test_labels.pbz2",
        # "df_dev.csv", "dev_rule_matches_z.pbz2", "dev_labels.pbz2"
    ]

    for file in tqdm(files):
        client.fget_object(
            bucket_name="knodle",
            object_name=os.path.join("datasets", dataset, "processed_unified_format", file),
            file_path=os.path.join(processed_data_dir, file),
        )

    max_features = 3800

    tfidf_data_module = KnodleDataModule(
        data_dir=processed_data_dir,
        batch_size=256,
        preprocessing_fct=get_tfidf_features,
        preprocessing_kwargs={"max_features": max_features},
        dataloader_train_keys=["x"]

    )
    tfidf_data_module.setup()


    num_classes = 6 if dataset == "trec" else 2
    train_metrics = {
        "accuracy": torchmetrics.Accuracy(num_classes=6)
    }

    test_metrics = {
        "accuracy": torchmetrics.Accuracy(),
        "f1": torchmetrics.F1(num_classes=6, average="macro")
    }

    model = LitModel(
        max_features=max_features, num_classes=6,
        train_metrics=train_metrics, test_metrics=test_metrics
    )

    trainer = LightningMajorityTrainer(max_epochs=20)
    # trainer = LightningSnorkelTrainer(max_epochs=20)
    trainer.fit(model=model, datamodule=k)



if __name__ == '__main__':
    main()