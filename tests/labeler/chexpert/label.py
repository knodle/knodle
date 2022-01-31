import bioc
import csv
import os
import time
import pandas as pd
from knodle.labeler.CheXpert.label import CheXpertLabeler
from knodle.labeler.CheXpert.preprocessing import Preprocessor
from tests.labeler.chexpert.config_tests import WeatherTestsConfig


CHEXPERT_DATA_DIR = os.path.join(os.getcwd())

# RULE DIRECTORIES -----------------------------------------------------------------------------------
MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
os.makedirs(MENTION_DATA_DIR, exist_ok=True)
with open(os.path.join(MENTION_DATA_DIR, 'storm.txt'), 'x') as f:
    f.write('storm')
    f.close()
with open(os.path.join(MENTION_DATA_DIR, 'clouds.txt'), 'x') as f:
    f.write('cloud')
    f.close()

UNMENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "unmention")
os.makedirs(UNMENTION_DATA_DIR, exist_ok=True)

# PATTERN DIRECTORY ----------------------------------------------------------------------------------
PATTERNS_DIR = os.path.join(CHEXPERT_DATA_DIR, "patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)
with open(os.path.join(PATTERNS_DIR, 'negation.txt'), 'w') as f:
    pass
with open(os.path.join(PATTERNS_DIR, 'pre_negation_uncertainty.txt'), 'w') as f:
    pass
with open(os.path.join(PATTERNS_DIR, 'post_negation_uncertainty.txt'), 'x') as f:
    f.write('{} < {} {lemma:/possible/}')
    f.close()

# SAMPLE DIRECTORY -----------------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(CHEXPERT_DATA_DIR, "reports")
os.makedirs(SAMPLE_DIR, exist_ok=True)
data = ['The weather is acting up today, even a storm is possible.']
with open(os.path.join(SAMPLE_DIR, 'countries.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(data)

# OUTPUT DIRECTORY -----------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(CHEXPERT_DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OTHER-----------------------------------------------------------------------------------------------
preprocessor = Preprocessor(config=WeatherTestsConfig())


def test_chexpert_labeler_label():

    labeler = CheXpertLabeler()

    labeler.label(uncertain=1, chexpert_bool=True)

    # Check that this runs without error
    assert True


def test_chexpert_labeler_clean():

    reports = pd.read_csv(os.path.join(SAMPLE_DIR, 'weather_forecast.csv'),
                          header=None,
                          names=[WeatherTestsConfig().reports])[WeatherTestsConfig().reports].tolist()
    for report in reports:
        assert preprocessor.clean(report) == "the weather is acting up today, even a storm is possible."


def test_chexpert_labeler_preprocess():

    preprocessor.preprocess()
    collection_new = preprocessor.collection

    part1 = "BioCCollection[source=,date=2022-01-31,key=,infons=[],documents=[BioCDocument[id=0,infons=[],"
    part2 = "passages=[BioCPassage[offset=0,text='the weather is ac ... torm is possible.',infons=[],"
    part3 = "sentences=[BioCSentence[offset=0,text='the weather is ac ... torm is possible.',"
    part4 = "infons=[],annotations=[],relations=[],]],annotations=[],relations=[],]],annotations=[],relations=[],]],]"

    collection_old = part1 + part2 + part3 + part4

    assert str(collection_new) == collection_old


# def test_chexpert_labeler_load_phrases():
#
#
