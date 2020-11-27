from joblib import dump
import logging
import os

logger = logging.getLogger(__name__)


def cache_data(file, path: str) -> None:
    folder_path = create_path_without_filename(path)
    create_folder(folder_path)
    dump(file, path)


def create_path_without_filename(full_path: str) -> str:
    path_without_filename = "/".join(full_path.split("/")[0:-1])
    return path_without_filename


def create_folder(folder_to_create):
    """
    Function which checks if a folder exists, if it doesn't -> Create it
    :param folder_to_create: Path to folder to check
    :return:
    """
    logger.info("Creating folder {}".format(folder_to_create))
    if not os.path.exists(folder_to_create):
        os.makedirs(folder_to_create)
