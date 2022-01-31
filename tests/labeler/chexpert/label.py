import csv
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from knodle.labeler.CheXpert.label import CheXpertLabeler
from knodle.labeler.CheXpert.preprocessing import Preprocessor
from knodle.labeler.CheXpert.matching import Matcher
from knodle.labeler.CheXpert.neg_unc_detection import NegUncDetector
from knodle.labeler.CheXpert.updating import Updater
from knodle.labeler.CheXpert.utils import z_matrix_fct, t_matrix_fct, get_rule_idx
from tests.labeler.chexpert.config_tests import WeatherTestsConfig


CHEXPERT_DATA_DIR = os.path.join(os.getcwd())

# RULE DIRECTORIES -----------------------------------------------------------------------------------
MENTION_DATA_DIR = os.path.join(CHEXPERT_DATA_DIR, "phrases", "mention")
os.makedirs(MENTION_DATA_DIR, exist_ok=True)
with open(os.path.join(MENTION_DATA_DIR, 'storm.txt'), 'w') as f:
    f.write('storm')
    f.close()
with open(os.path.join(MENTION_DATA_DIR, 'clouds.txt'), 'w') as f:
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
with open(os.path.join(PATTERNS_DIR, 'post_negation_uncertainty.txt'), 'w') as f:
    f.write('{} < {} {lemma:/possible/}')
    f.close()

# SAMPLE DIRECTORY -----------------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(CHEXPERT_DATA_DIR, "reports")
os.makedirs(SAMPLE_DIR, exist_ok=True)
data = ['The weather is acting up today, even a storm is possible.']
with open(os.path.join(SAMPLE_DIR, 'weather_forecast.csv.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(data)

# OUTPUT DIRECTORY -----------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(CHEXPERT_DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OTHER-----------------------------------------------------------------------------------------------
preprocessor = Preprocessor(config=WeatherTestsConfig())
matcher = Matcher(config=WeatherTestsConfig(), chexpert_data=False)
neg_unc_detector = NegUncDetector(config=WeatherTestsConfig())
updater = Updater(config=WeatherTestsConfig())
labeler = CheXpertLabeler(labeler_config=WeatherTestsConfig())


# TESTS: "LABEL"--------------------------------------------------------------------------------------
def test_chexpert_labeler_label():

    labeler.label(uncertain=1, chexpert_bool=False)

    # Check that this runs without error
    assert True


def test_chexpert_labeler_label_x_matrix():

    labeler.label(uncertain=-1, chexpert_bool=False)
    x_matrix = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_matrix.csv'), header=None).to_numpy()

    x_matrix_expected = np.array([['The weather is acting up today, even a storm is possible.']])

    np.testing.assert_equal(x_matrix, x_matrix_expected)


def test_chexpert_labeler_label_t_matrix():

    labeler.label(uncertain=-1, chexpert_bool=False)
    t_matrix = pd.read_csv(os.path.join(OUTPUT_DIR, 'T_matrix.csv'), header=None).to_numpy()

    t_matrix_expected = np.array([[1, 0], [0, 1]], dtype=float)

    np.testing.assert_equal(t_matrix, t_matrix_expected)


def test_chexpert_labeler_label_z_matrix():

    labeler.label(uncertain=-1, chexpert_bool=False)
    z_matrix = pd.read_csv(os.path.join(OUTPUT_DIR, 'Z_matrix.csv'), header=None).to_numpy()

    z_matrix_expected = np.array([[0, -1]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "PREPROCESSING"------------------------------------------------------------------------------
def test_chexpert_labeler_clean():

    reports = pd.read_csv(os.path.join(SAMPLE_DIR, 'weather_forecast.csv'),
                          header=None,
                          names=[WeatherTestsConfig().reports])[WeatherTestsConfig().reports].tolist()

    for report in reports:
        assert preprocessor.clean(report) == "the weather is acting up today, even a storm is possible."


def test_chexpert_labeler_preprocess():

    preprocessor.preprocess()
    collection = preprocessor.collection

    part1 = "BioCCollection[source=,date="
    date = time.strftime("%Y-%m-%d")  # get date of today
    part2 = ",key=,infons=[],documents=[BioCDocument[id=0,infons=[],"
    part3 = "passages=[BioCPassage[offset=0,text='the weather is ac ... torm is possible.',infons=[],"
    part4 = "sentences=[BioCSentence[offset=0,text='the weather is ac ... torm is possible.',"
    part5 = "infons=[],annotations=[],relations=[],]],annotations=[],relations=[],]],annotations=[],relations=[],]],]"
    collection_expected = part1 + date + part2 + part3 + part4 + part5

    assert str(collection) == collection_expected


# TESTS: "MATCHING"-----------------------------------------------------------------------------------
def test_chexpert_labeler_load_phrases():

    observation2mention_phrases = matcher.observation2mention_phrases

    observation2mention_phrases_expected = defaultdict(list)
    observation2mention_phrases_expected['Clouds.Txt'].append('cloud')
    observation2mention_phrases_expected['Storm.Txt'].append('storm')

    assert observation2mention_phrases == observation2mention_phrases_expected


def test_chexpert_labeler_match():

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)

    part1 = "BioCCollection[source=,date="
    date = time.strftime("%Y-%m-%d")  # get date of today
    part2 = ",key=,infons=[],documents=[BioCDocument[id=0,infons=[],"
    part3 = "passages=[BioCPassage[offset=0,text='the weather is ac ... torm is possible.',infons=[],"
    part4 = "sentences=[BioCSentence[offset=0,text='the weather is ac ... torm is possible.',"
    part5 = "infons=[],annotations=[],relations=[],]],annotations=[BioCAnnotation[id=0,text='storm',"
    part6 = "infons=[term=storm,observation=Storm.Txt,annotator=Phrase],"
    part7 = "locations=[BioCLocation[offset=39,length=5]],]],relations=[],]],annotations=[],relations=[],]],]"
    collection_expected = part1 + date + part2 + part3 + part4 + part5 + part6 + part7

    assert str(collection) == collection_expected


def test_chexpert_labeler_match_z_matrix():

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)
    z_matrix = matcher.z_matrix

    z_matrix_expected = np.array([[0, 999]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "NEG_UNC_DETECTION"--------------------------------------------------------------------------
def test_chexpert_labeler_neg_unc_detect():

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)
    neg_unc_detector.neg_unc_detect(collection)

    part1 = "BioCCollection[source=,date="
    date = time.strftime("%Y-%m-%d")  # get date of today
    part2 = ",key=,infons=[],documents=[BioCDocument[id=0,infons=[],"
    part3 = "passages=[BioCPassage[offset=0,text='the weather is ac ... torm is possible.',infons=[],"
    part4 = "sentences=[],annotations=[BioCAnnotation[id=0,text='storm',"
    part5 = "infons=[term=storm,observation=Storm.Txt,annotator=Phrase,uncertainty=True],"
    part6 = "locations=[BioCLocation[offset=39,length=5]],]],relations=[],]],annotations=[],relations=[],]],]"
    collection_expected = part1 + date + part2 + part3 + part4 + part5 + part6

    assert str(collection) == collection_expected


# TESTS: "UPDATING"-----------------------------------------------------------------------------------
def test_chexpert_labeler_update():

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)
    neg_unc_detector.neg_unc_detect(collection)
    z_matrix = updater.update(collection, matcher.z_matrix, chexpert_data=False)

    z_matrix_expected = np.array([[0, -1]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "UTILS"--------------------------------------------------------------------------------------
def test_chexpert_labeler_t_matrix_fct():

    t_matrix = t_matrix_fct(config=WeatherTestsConfig())

    t_matrix_expected = np.array([[1, 0], [0, 1]])

    np.testing.assert_equal(t_matrix, t_matrix_expected)


def test_chexpert_labeler_z_matrix_fct():

    z_matrix = z_matrix_fct(config=WeatherTestsConfig())

    z_matrix_expected = np.zeros((1, 2))

    np.testing.assert_equal(z_matrix, z_matrix_expected)


def test_chexpert_labeler_get_rule_idx():

    idx = get_rule_idx(phrase="cloud", config=WeatherTestsConfig())

    idx_expected = pd.Int64Index([0])

    pd.testing.assert_index_equal(idx, idx_expected)
