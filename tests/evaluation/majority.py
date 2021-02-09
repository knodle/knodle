import numpy as np

from knodle.evaluation.majority import majority_sklearn_report


def test_majority_vote_no_match():
    z = np.zeros((2, 4))
    t = np.zeros((4, 2))

    z[0, 0] = 1
    z[1, 1] = 1

    t[0, 0] = 1
    t[1, 0] = 1
    t[2, 0] = 1
    t[3, 0] = 1

    y = np.array([1, 1])
    report = majority_sklearn_report(z, t, y)
    assert report["accuracy"] == 0


def test_majority_vote_correctness():
    z = np.zeros((2, 4))
    t = np.zeros((4, 2))

    z[0, 0] = 1
    z[1, 1] = 1

    t[0, 0] = 1
    t[1, 1] = 1
    t[2, 0] = 1
    t[3, 0] = 1

    y = np.array([1, 1])
    report = majority_sklearn_report(z, t, y)
    assert report["accuracy"] == 0.5

    y = np.array([0, 1])
    report = majority_sklearn_report(z, t, y)
    assert report["accuracy"] == 1


def test_majority_vote_random_guess():
    z = np.zeros((4, 4))
    t = np.zeros((4, 2))

    z[0, 0] = 1
    z[1, 1] = 1
    z[2, 0] = 1

    t[0, 0] = 1
    t[1, 1] = 1
    t[2, 0] = 1
    t[3, 0] = 1

    # no info about sample 4, thus random value is chosen
    # y_majority = np.array([0, 1, 0, random])
    y = np.array([0, 1, 1, 1])
    report = majority_sklearn_report(z, t, y)
    assert 0.5 <= report["accuracy"] <= 0.75
