from pathlib import Path
import os
from minio import Minio

# Paths
HOME_DIR = Path.home()
PARSING_MODEL_DIR = HOME_DIR / ".local/share/bllipparser/GENIA+PubMed"

CHEXPERT_DATA_DIR = os.path.join(os.getcwd(), "example", "chexpert")

MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
# os.makedirs(MENTION_DATA_DIR, exist_ok=True)

UNMENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "unmention")
# os.makedirs(UNMENTION_DATA_DIR, exist_ok=True)

REPORTS_PATH = os.path.join(CHEXPERT_DATA_DIR, "reports", "sample_reports.csv")
#os.makedirs(REPORTS_PATH, exist_ok=True)

PRE_NEG_UNC_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "pre_negation_uncertainty.txt")
NEG_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "negation.txt")
POST_NEG_UNC_PATH = os.path.join(CHEXPERT_DATA_DIR, "patterns", "post_negation_uncertainty.txt")



# Download data
#CLIENT = Minio("knodle.cc", secure=False)
FILES = os.listdir(MENTION_DATA_DIR)
FILES.sort()




# Observation constants
CARDIOMEGALY = "Cardiomegaly"
ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
SUPPORT_DEVICES = "Support Devices"
NO_FINDING = "No Finding"
OBSERVATION = "observation"
CATEGORIES = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
              "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices"]

# Numeric constants
POSITIVE = 1
NEGATIVE = 0
UNCERTAIN = -1

# Misc. constants
UNCERTAINTY = "uncertainty"
NEGATION = "negation"
REPORTS = "Reports"
