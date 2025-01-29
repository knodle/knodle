from snorkel.labeling.model import LabelModel
from knodle.trainer.snorkel.utils import z_t_matrix_to_snorkel_matrix, prepare_empty_rule_matches

from knodle.trainer.lightning.trainer import KnodleLightningLabelModelTrainer


class LightningSnorkelTrainer(KnodleLightningLabelModelTrainer):
    def __init__(
            self,
            use_probabilistic_labels: bool = False,
            label_model_num_epochs: int = 5000,
            label_model_log_freq: int = 500,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.use_probabilistic_labels = use_probabilistic_labels
        self.probability_threshold = None
        self.label_model_num_epochs = label_model_num_epochs
        self.label_model_log_freq = label_model_log_freq

    def apply_snorkel_denoising(
            self, datamodule
    ):
        # create Snorkel matrix
        non_empty_mask, rule_matches_z = prepare_empty_rule_matches(datamodule.data["train_rule_matches_z"])
        L_train = z_t_matrix_to_snorkel_matrix(rule_matches_z, datamodule.data["mapping_rules_labels_t"])

        # train LabelModel
        label_model = LabelModel(cardinality=datamodule.data["mapping_rules_labels_t"].shape[1], verbose=True)
        fitting_kwargs = {}
        # TODO seeding; Lightning seed all?
        # if seed is not None:
        #     fitting_kwargs["seed"] = self.trainer_config.seed

        label_model.fit(
            L_train,
            n_epochs=self.label_model_num_epochs,
            log_freq=self.label_model_log_freq,
            **fitting_kwargs
        )
        label_probs_gen = label_model.predict_proba(L_train)

        datamodule.data["train_rule_matches_z"] = datamodule.data["train_rule_matches_z"][non_empty_mask]
        datamodule.data["train_weak_y"] = label_probs_gen.argmax(axis=-1)

        for key in datamodule.dataloader_train_keys:
            datamodule.data[f"train_{key}"] = datamodule.data[f"train_{key}"][non_empty_mask]

        assert datamodule.data["train_weak_y"].shape[0] == datamodule.data["train_rule_matches_z"].shape[0]
        return datamodule

    def apply_label_model(self, datamodule):
        return self.apply_snorkel_denoising(datamodule)
