import os

from minio import Minio
from tqdm.auto import tqdm


class MinioConnector:
    def __init__(self):
        self.client = Minio(
            "knodle.dm.univie.ac.at",
            secure=False
        )
        self.bucket = "knodle"

    def list_files(self, path: str, recursive: bool = True):
        path = normalize_dir_path(path)

        objects = self.client.list_objects(self.bucket, prefix=path, recursive=recursive)
        files = [obj._object_name.replace(path, '') for obj in objects]

        return files

    def download_file(self, source_path: str, target_path: str):
        self.client.fget_object(
            bucket_name=self.bucket,
            object_name=source_path,
            file_path=target_path
        )

    def download_dir(self, source_folder: str, target_folder: str, recursive: bool = True):
        source_folder = normalize_dir_path(source_folder)
        target_folder = normalize_dir_path(target_folder)
        files = self.list_files(source_folder, recursive=recursive)

        os.makedirs(target_folder, exist_ok=True)
        for file in tqdm(files):
            source_file = f"{source_folder}{file}"
            target_file = f"{target_folder}{file}"
            self.download_file(source_file, target_file)


def normalize_dir_path(path: str) -> str:
    if path.endswith("/"):
        path = f"{path}/"
    return path
