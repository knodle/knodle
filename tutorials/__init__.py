"""knodle - Knowledge infused deep learning framework"""


__version__ = "0.1.0"
__author__ = "knodle <knodle@cs.univie.ac.at>"
__all__ = []

import logging

logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

file_log_handler = logging.FileHandler('logfile.log')
file_log_handler.setFormatter(formatter)
logger.addHandler(file_log_handler)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("Initalized logger")
