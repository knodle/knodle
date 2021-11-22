import pathlib
from typing import Callable, Dict
import os
import logging

import torch
from torch import Tensor

from knodle.trainer.utils.utils import check_and_return_device, set_seed

logger = logging.getLogger(__name__)


class LabelerConfig:
    def __init__(
            self,
            caching_folder: str = os.path.join(pathlib.Path().absolute(), "cache"),
            caching_suffix: str = "",
            labeled_reports_dir: str = None
    ):
        self.caching_suffix = caching_suffix
        self.caching_folder = caching_folder
        os.makedirs(self.caching_folder, exist_ok=True)
        logger.info(f"The cache will be saved to {self.caching_folder} folder")

        # create directory where labeled reports will be stored
        if labeled_reports_dir:
            self.labeled_reports_dir = labeled_reports_dir
            os.makedirs(self.labeled_reports_dir, exist_ok=True)
        else:
            self.labeled_reports_dir = caching_folder
        logger.info(f"The labeled reports will be saved to the {self.labeled_reports_dir} directory.")


class BaseLabelerConfig(LabelerConfig):
    def __init__(
            self,
            filter_non_labelled: bool = True,
            other_class_id: int = None,
            evaluate_with_other_class: bool = False,
            ids2labels: Dict = None,
            max_rules: int = None,
            min_coverage: float = None,
            drop_rules: bool = False,
            **kwargs
    ):
        """
        Additionally provided parameters needed for handling the cases where there are data samples with no rule
        matched (filtering OR introducing the other class + training & evaluation with other class).

        :param filter_non_labelled: if True, the samples with no rule matched will be filtered out from the dataset
        :param other_class_id: id of the negative class; if set, the samples with no rule matched will be assigned to it
        :param evaluate_with_other_class: if set to True, the evaluation will be done with respect to the negative class
        (for more details please see knodle/evaluation/other_class_metrics.py file)
        :param ids2labels: dictionary {label id: label}, which is needed to perform evaluation with the negative class
        """
        super().__init__(**kwargs)
        self.filter_non_labelled = filter_non_labelled
        self.other_class_id = other_class_id
        self.evaluate_with_other_class = evaluate_with_other_class
        self.ids2labels = ids2labels

        if self.other_class_id is not None and self.filter_non_labelled:
            raise ValueError("You can either filter samples with no weak labels or add them to 'other_class_id'")

        logger.debug(f"{self.evaluate_with_other_class} and {self.ids2labels}")
        if self.evaluate_with_other_class and self.ids2labels is None:
            # check if the selected evaluation type is valid
            logging.warning(
                "Labels to label ids correspondence is needed to make other_class specific evaluation. Since it is "
                "absent now, the standard sklearn metrics will be calculated instead."
            )
            self.evaluate_with_other_class = False

        self.max_rules = max_rules
        self.min_coverage = min_coverage
        self.drop_rules = drop_rules
