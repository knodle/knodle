from knodle.labeler.CheXpert.label import CheXpertLabeler
from knodle.labeler.CheXpert.config import LabelerConfig


def test_label():

    labeler = CheXpertLabeler()

    labeler.label(transform_patterns=False, uncertain=1, chexpert_bool=True)

    # Check that this runs without error
    assert True
