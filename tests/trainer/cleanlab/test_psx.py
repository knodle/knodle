import numpy as np
# todo: test compute psx matrix -> check also that there is no duplicates
#  psx[indices_holdout_cv] = psx_cv : value for each sample is filled only once
# for n, idx in enumerate(indices_holdout_cv):
#     if psx[idx][0] == 0 and psx[idx][1] == 0:
#         psx[idx, :] = psx_cv[n, :]
#     else:
#         logging.info("ALARMAAAA")

from knodle.trainer.cleanlab.utils import calculate_threshold


def test_threshold_calculation():
    psx = np.array([[1., 2.], [5., 6.], [4., 3.]])
    labels = np.array([1., 1., 0.])
    true_threshold = np.array([4., 4.])

    threshold = calculate_threshold(psx, labels)

    assert np.array_equal(threshold, true_threshold)
