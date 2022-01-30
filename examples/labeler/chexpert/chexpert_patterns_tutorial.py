import os
from tqdm.auto import tqdm
from minio import Minio
from knodle.labeler.CheXpert.label import CheXpertLabeler
from examples.labeler.chexpert.config_pattern_tutorial import WeatherConfig

# This notebook demonstrates how negation & uncertainty patterns for the CheXpert labeler work.
# Please have a look at the identically named jupyter notebook for further explanations.

# Define Constants & Download Data.

client = Minio("knodle.cc", secure=False)

# Directory where all the files should be saved in is specified.
CHEXPERT_DATA_DIR = os.path.join(os.getcwd(), "examples", "labeler", "chexpert")


# RULE DIRECTORIES -----------------------------------------------------------------------------------
MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
os.makedirs(MENTION_DATA_DIR, exist_ok=True)
files_mention = [
    "clouds.txt", "cold.txt", "rain.txt", "snow.txt",
    "storm.txt", "sun.txt", "warm.txt", "wind.txt"
]
for file in tqdm(files_mention):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/weather/phrases/mention/", file),
        file_path=os.path.join(MENTION_DATA_DIR, file),
    )

UNMENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "unmention")
os.makedirs(UNMENTION_DATA_DIR, exist_ok=True)
files_unmention = [
    "rain.txt"
]
for file in tqdm(files_unmention):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/weather/phrases/unmention/", file),
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
        object_name=os.path.join("datasets/weather/patterns/", file),
        file_path=os.path.join(PATTERNS_DIR, file),
    )


# SAMPLE DIRECTORY -----------------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(CHEXPERT_DATA_DIR, "reports")
os.makedirs(SAMPLE_DIR, exist_ok=True)
files_sample = [
    "weather_forecast.csv"
]
for file in tqdm(files_sample):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/weather/reports/", file),
        file_path=os.path.join(SAMPLE_DIR, file),
    )


# OUTPUT DIRECTORY -----------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(CHEXPERT_DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# The labeler class is initiated using the "config_pattern_tutorial.py" as config file.
labeler = CheXpertLabeler(labeler_config=WeatherConfig())

# The label function is run, outputting the matrices X, T and Z.
labeler.label(uncertain=-1, chexpert_bool=False)
