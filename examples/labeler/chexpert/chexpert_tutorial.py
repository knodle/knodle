import os
from tqdm.auto import tqdm
from minio import Minio
from knodle.labeler.CheXpert.label import Labeler

# Define Constants & Download Data
#
# In this tutorial all data is downloaded from the knodle Minio browser & saved in
# the knodle.examples.labeler.chexpert directory, in the default config.py file, these paths are accessed.

client = Minio("knodle.cc", secure=False)

# Directory where all the files should be saved in is specified
CHEXPERT_DATA_DIR = os.path.join(os.getcwd(), "examples", "labeler", "chexpert")


# RULE DIRECTORIES -----------------------------------------------------------------------------------
MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
os.makedirs(MENTION_DATA_DIR, exist_ok=True)
files_mention = [
    "pleural_effusion.txt", "no_finding.txt", "edema.txt", "support_devices.txt", "lung_lesion.txt",
    "cardiomegaly.txt", "pneumothorax.txt", "atelectasis.txt", "fracture.txt", "pneumonia.txt",
    "pleural_other.txt", "consolidation.txt", "enlarged_cardiomediastinum.txt", "lung_opacity.txt"
]
for file in tqdm(files_mention):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("examples/labeler/chexpert/phrases/mention/", file),
        file_path=os.path.join(MENTION_DATA_DIR, file),
    )

UNMENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "unmention")
os.makedirs(UNMENTION_DATA_DIR, exist_ok=True)
files_unmention = [
    "pleural_effusion.txt", "lung_opacity.txt", "lung_lesion.txt"
]
for file in tqdm(files_unmention):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("examples/labeler/chexpert/phrases/unmention/", file),
        file_path=os.path.join(UNMENTION_DATA_DIR, file),
    )


# PATTERN DIRECTORY ----------------------------------------------------------------------------------
PATTERNS_DIR = os.path.join(CHEXPERT_DATA_DIR, "patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)
files_patterns = [
    "pre_negation_uncertainty.txt", "negation.txt", "post_negation_uncertainty.txt"
]
for file in tqdm(files_patterns):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("examples/labeler/chexpert/patterns/", file),
        file_path=os.path.join(PATTERNS_DIR, file),
    )


# SAMPLE DIRECTORY -----------------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(CHEXPERT_DATA_DIR, "reports")
os.makedirs(SAMPLE_DIR, exist_ok=True)
files_sample = [
    "sample_reports.csv"
]
for file in tqdm(files_sample):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("examples/labeler/chexpert/reports/", file),
        file_path=os.path.join(SAMPLE_DIR, file),
    )


# OUTPUT DIRECTORY -----------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(CHEXPERT_DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# The labeler class is initiated without passing a config.py file, so the default one is used.
labeler = Labeler()

# The label function is run, outputting the matrices X, T and Z.
labeler.label(transform_patterns=False, uncertain=1, chexpert_bool=True)
