import numpy as np

from knodle.trainer.snorkel.snorkel_trainer import SnorkelTrainer
from knodle.trainer.snorkel.config import SnorkelConfig


def test_snorkel_with_other_class():
    z = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])

    t = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    config = SnorkelConfig(
        optimizer=3,
        other_class_id=2,
        filter_non_labelled=False
    )
    trainer = SnorkelTrainer(
        model=None,
        rule_matches_z=z,
        mapping_rules_labels_t=t,
        model_input_x=None,
        trainer_config=config
    )

    snorkel_gold = np.array([
        2, 1, 1, 2, 0, 2
    ])

    _, label_probs = trainer._snorkel_denoising()
    np.testing.assert_equal(label_probs.argmax(axis=1), snorkel_gold)
