import os
from tqdm.auto import tqdm
from minio import Minio

# Define Constants & Download Data
#
# In this tutorial all data is downloaded from the knodle Minio browser
# For the CheXpert labeler to work, all the constants below which are written in caps need to be specified in the
# config.py file, if

client = Minio("knodle.cc", secure=False)
CHEXPERT_DATA_DIR = os.path.join(os.getcwd(), "examples", "labeler", "chexpert")

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
PRE_NEG_UNC_PATH = os.path.join(PATTERNS_DIR, "pre_negation_uncertainty.txt")
NEG_PATH = os.path.join(PATTERNS_DIR, "negation.txt")
POST_NEG_UNC_PATH = os.path.join(PATTERNS_DIR, "post_negation_uncertainty.txt")

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

OUTPUT_DIR = os.path.join(CHEXPERT_DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all the mention files and sort them alphabetically to avoid undesired behaviour when T-matrix is created
FILES = os.listdir(MENTION_DATA_DIR)
FILES.sort()

# Observation constants - CheXpert data specific
CARDIOMEGALY = "Cardiomegaly"
ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
SUPPORT_DEVICES = "Support Devices"
NO_FINDING = "No Finding"
OBSERVATION = "observation"


from knodle.labeler.CheXpert.label import label
label(transform_patterns=False, uncertain=1, chexpert_bool=True)
