"""knodle - Knowledge infused deep learning framework"""


__version__ = "0.1.0"
__author__ = "knodle <knodle@cs.univie.ac.at>"
__all__ = []

import logging

# from knodle.model.logistic_regression.logistic_regression_model import (
#    LogisticRegressionModel,
# )

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("Initalized logger")
