"""knodle - Knowledge infused deep learning framework"""

__version__ = "0.1.0"
__author__ = "knodle <knodle@cs.univie.ac.at>"
__all__ = []

import logging
import sys

logger = logging.getLogger()
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("Initalized logger")
