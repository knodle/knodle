from pathlib import Path
import os
import pandas as pd
import numpy as np


# Paths
HOME_DIR = Path.home()
PARSING_MODEL_DIR = HOME_DIR / ".local/share/bllipparser/GENIA+PubMed"

CHEXPERT_DATA_DIR = os.path.join(os.getcwd(), "examples", "labeler", "chexpert")

MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
UNMENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "unmention")

REPORTS_PATH = os.path.join(CHEXPERT_DATA_DIR, "reports", "sample_reports.csv")

OUTPUT_PATH = os.path.join(CHEXPERT_DATA_DIR, "output")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

PRE_NEG_UNC_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "pre_negation_uncertainty.txt")
NEG_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "negation.txt")
POST_NEG_UNC_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "post_negation_uncertainty.txt")

# Get all the mention files and sort them alphabetically to avoid undesired behaviour when T-matrix is created
FILES = os.listdir(MENTION_DATA_DIR)
FILES.sort()

# Observation constants - CheXpert data specific
CARDIOMEGALY = "Cardiomegaly"
ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
SUPPORT_DEVICES = "Support Devices"
NO_FINDING = "No Finding"
OBSERVATION = "observation"

# Numeric constants
POSITIVE = 1
NEGATIVE = 0
UNCERTAIN = -1

# Misc. constants
UNCERTAINTY = "uncertainty"
NEGATION = "negation"
REPORTS = "Reports"
