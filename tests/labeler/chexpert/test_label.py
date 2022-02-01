import csv
import os
import pytest
import time

import numpy as np
import pandas as pd

from collections import defaultdict

from tests.labeler.chexpert.config_test_label import WeatherTestsConfig

from knodle.labeler.CheXpert.label import CheXpertLabeler
from knodle.labeler.CheXpert.preprocessing import Preprocessor
from knodle.labeler.CheXpert.matching import Matcher
from knodle.labeler.CheXpert.neg_unc_detection import NegUncDetector
from knodle.labeler.CheXpert.updating import Updater
from knodle.labeler.CheXpert.utils import z_matrix_fct, t_matrix_fct, get_rule_idx


@pytest.fixture
def directories():

    chexpert_data_dir = os.path.join(os.getcwd())

    # RULE DIRECTORIES -----------------------------------------------------------------------------------
    mention_data_dir = os.path.join(chexpert_data_dir, "phrases", "mention")
    os.makedirs(mention_data_dir, exist_ok=True)
    with open(os.path.join(mention_data_dir, 'storm.txt'), 'w') as f:
        f.write('storm')
        f.close()
    with open(os.path.join(mention_data_dir, 'clouds.txt'), 'w') as f:
        f.write('cloud')
        f.close()
    with open(os.path.join(mention_data_dir, 'rain.txt'), 'w') as f:
        f.write('rain\n')
        f.write('pour\n')
        f.write('drizzle')
        f.close()

    unmention_data_dir = os.path.join(chexpert_data_dir, "phrases", "unmention")
    os.makedirs(unmention_data_dir, exist_ok=True)
    with open(os.path.join(unmention_data_dir, 'rain.txt'), 'w') as f:
        f.write('dry')
        f.close()

    # PATTERN DIRECTORY ----------------------------------------------------------------------------------
    patterns_dir = os.path.join(chexpert_data_dir, "patterns")
    os.makedirs(patterns_dir, exist_ok=True)
    with open(os.path.join(patterns_dir, 'negation.txt'), 'w') as f:
        pass
    with open(os.path.join(patterns_dir, 'pre_negation_uncertainty.txt'), 'w') as f:
        pass
    with open(os.path.join(patterns_dir, 'post_negation_uncertainty.txt'), 'w') as f:
        f.write('{} < {} {lemma:/possible/}')
        f.close()

    # SAMPLE DIRECTORY -----------------------------------------------------------------------------------
    sample_dir = os.path.join(chexpert_data_dir, "reports")
    os.makedirs(sample_dir, exist_ok=True)
    data = ['The weather is acting up today, even a storm is possible.']
    with open(os.path.join(sample_dir, 'weather_forecast.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    # OUTPUT DIRECTORY -----------------------------------------------------------------------------------
    output_dir = os.path.join(chexpert_data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    return sample_dir, output_dir


@pytest.fixture
def classes():

    preprocessor = Preprocessor(config=WeatherTestsConfig())
    matcher = Matcher(config=WeatherTestsConfig(), chexpert_data=False)
    neg_unc_detector = NegUncDetector(config=WeatherTestsConfig())
    updater = Updater(config=WeatherTestsConfig())

    return preprocessor, matcher, neg_unc_detector, updater


# TESTS: "LABEL"--------------------------------------------------------------------------------------
def test_chexpert_labeler_label(directories):

    directories()

    labeler = CheXpertLabeler(labeler_config=WeatherTestsConfig())

    labeler.label(uncertain=1, chexpert_bool=False)

    # Check that this runs without error
    assert True


def test_chexpert_labeler_label_x_matrix(directories):

    _, output_dir = directories()

    labeler = CheXpertLabeler(labeler_config=WeatherTestsConfig())

    labeler.label(uncertain=-1, chexpert_bool=False)
    x_matrix = pd.read_csv(os.path.join(output_dir, 'X_matrix.csv'), header=None).to_numpy()

    x_matrix_expected = np.array([['The weather is acting up today, even a storm is possible.']])

    np.testing.assert_equal(x_matrix, x_matrix_expected)


def test_chexpert_labeler_label_t_matrix(directories):

    _, output_dir = directories()

    labeler = CheXpertLabeler(labeler_config=WeatherTestsConfig())

    labeler.label(uncertain=-1, chexpert_bool=False)
    t_matrix = pd.read_csv(os.path.join(output_dir, 'T_matrix.csv'), header=None).to_numpy()

    t_matrix_expected = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]], dtype=float)

    np.testing.assert_equal(t_matrix, t_matrix_expected)


def test_chexpert_labeler_label_z_matrix(directories):

    _, output_dir = directories()

    labeler = CheXpertLabeler(labeler_config=WeatherTestsConfig())

    labeler.label(uncertain=-1, chexpert_bool=False)
    z_matrix = pd.read_csv(os.path.join(output_dir, 'Z_matrix.csv'), header=None).to_numpy()

    z_matrix_expected = np.array([[0, 0, 0, 0, -1]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "PREPROCESSING"------------------------------------------------------------------------------
def test_chexpert_labeler_clean(directories, classes):

    sample_dir, _ = directories()

    preprocessor, _, _, _ = classes

    reports = pd.read_csv(os.path.join(sample_dir, 'weather_forecast.csv'),
                          header=None,
                          names=[WeatherTestsConfig().reports])[WeatherTestsConfig().reports].tolist()

    for report in reports:
        assert preprocessor.clean(report) == "the weather is acting up today, even a storm is possible."


def test_chexpert_labeler_preprocess(directories, classes):

    directories()

    preprocessor, _, _, _ = classes

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
def test_chexpert_labeler_load_phrases(directories, classes):

    directories()

    _, matcher, _, _ = classes

    observation2mention_phrases = matcher.observation2mention_phrases

    observation2mention_phrases_expected = defaultdict(list)
    observation2mention_phrases_expected['Clouds.Txt'].append('cloud')
    observation2mention_phrases_expected['Rain.Txt'].append('rain')
    observation2mention_phrases_expected['Rain.Txt'].append('pour')
    observation2mention_phrases_expected['Rain.Txt'].append('drizzle')
    observation2mention_phrases_expected['Storm.Txt'].append('storm')

    assert observation2mention_phrases == observation2mention_phrases_expected


def test_chexpert_labeler_match(directories, classes):

    directories()

    preprocessor, matcher, _, _ = classes

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


def test_chexpert_labeler_match_z_matrix(directories, classes):

    directories()

    preprocessor, matcher, _, _ = classes

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)
    z_matrix = matcher.z_matrix

    z_matrix_expected = np.array([[0, 0, 0, 0, 999]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "NEG_UNC_DETECTION"--------------------------------------------------------------------------
def test_chexpert_labeler_neg_unc_detect(directories, classes):

    directories()

    preprocessor, matcher, neg_unc_detector, _ = classes

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
def test_chexpert_labeler_update(directories, classes):

    directories()

    preprocessor, matcher, neg_unc_detector, updater = classes

    preprocessor.preprocess()
    collection = preprocessor.collection
    matcher.match(collection)
    neg_unc_detector.neg_unc_detect(collection)
    z_matrix = updater.update(collection, matcher.z_matrix, chexpert_data=False)

    z_matrix_expected = np.array([[0, 0, 0, 0, -1]], dtype=float)

    np.testing.assert_equal(z_matrix, z_matrix_expected)


# TESTS: "UTILS"--------------------------------------------------------------------------------------
def test_chexpert_labeler_t_matrix_fct(directories):

    directories()

    t_matrix = t_matrix_fct(config=WeatherTestsConfig())

    t_matrix_expected = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

    np.testing.assert_equal(t_matrix, t_matrix_expected)


def test_chexpert_labeler_z_matrix_fct(directories):

    directories()

    z_matrix = z_matrix_fct(config=WeatherTestsConfig())

    z_matrix_expected = np.zeros((1, 5))

    np.testing.assert_equal(z_matrix, z_matrix_expected)


def test_chexpert_labeler_get_rule_idx(directories):

    directories()

    idx = get_rule_idx(phrase="drizzle", config=WeatherTestsConfig())

    idx_expected = pd.Int64Index([3])

    pd.testing.assert_index_equal(idx, idx_expected)
