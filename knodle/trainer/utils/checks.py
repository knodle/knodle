import logging

from numpy import ndarray
from knodle.trainer.config import BaseTrainerConfig


def check_other_class_id(trainer_config: BaseTrainerConfig, mapping_rules_labels_t: ndarray):
    # check and derive other_class_id from class mappings if neccessary
    if trainer_config.other_class_id is None:
        if not trainer_config.filter_non_labelled:
            trainer_config.other_class_id = mapping_rules_labels_t.shape[1]
    elif trainer_config.other_class_id < 0:
        raise RuntimeError("Label for negative samples should be greater than 0 for correct matrix multiplication")
    elif trainer_config.other_class_id < mapping_rules_labels_t.shape[1] - 1:
        logging.warning(f"Negative class {trainer_config.other_class_id} is already present in data")
