import pytest
from data_preprocessing_wiener_diarum_toponym import create_labels

def test_create_labels():
    example_sentence = ["Die Anna Musterfrau 99 J. gestorben in der <mark>Zucker/warenfabrik Wonka</mark> in der <mark>Königsegggasse 7</mark> an einer Zuckerüberdosis.","Dieser Satz hat leider nur einen halben <mark>Ortsnamen daher kann er nicht verwendet werden."]
    example_cleaned = [["Die", "Anna", "Musterfrau", "99", "J.", "gestorben", "in", "der", "Zuckerwarenfabrik", "Wonka ", "in", "der", "Königsegggasse", "7", "an", "einer", "Zuckerüberdosis."]]
    example_label = [[None, None, None, None, None, None, None, None, None, 0, 0, None, None, 1, 1, None, None, None]]
    example_place_name_dic = {"Zuckerwarenfabrik Wonka":0, "Königsegggasse 7": 1}
    assert create_labels(example_sentence) == example_label,example_cleaned,example_place_name_dic


