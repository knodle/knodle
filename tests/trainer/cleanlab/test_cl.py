from torch.nn import CrossEntropyLoss

from tests.trainer.generic import std_trainer_input_2

from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def test_cleanlab_base_test(std_trainer_input_2):
    (
        model,
        inputs_x, mapping_rules_labels_t, train_rule_matches_z,
        test_dataset, test_labels
    ) = std_trainer_input_2

    config = CleanLabConfig(cv_n_folds=2, criterion=CrossEntropyLoss, use_probabilistic_labels=False)

    trainer = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True
