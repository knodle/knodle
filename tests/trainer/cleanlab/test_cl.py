from torch.nn import CrossEntropyLoss

from tests.trainer.generic import std_trainer_input_2

from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def test_cleanlab_base_test_random(std_trainer_input_2):
    (
        model,
        inputs_x,
        mapping_rules_labels_t,
        train_rule_matches_z,
        test_dataset,
        test_labels
    ) = std_trainer_input_2

    config_random = CleanLabConfig(cv_n_folds=2, criterion=CrossEntropyLoss, psx_calculation_method="random", seed=1234)

    trainer_random = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config_random
    )

    trainer_random.train()

    _ = trainer_random.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True


def test_cleanlab_base_test_rules(std_trainer_input_2):
    (
        model,
        inputs_x,
        mapping_rules_labels_t,
        train_rule_matches_z,
        test_dataset,
        test_labels
    ) = std_trainer_input_2

    config_rules = CleanLabConfig(cv_n_folds=2, criterion=CrossEntropyLoss, psx_calculation_method="rules", seed=1234)

    trainer_rules = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config_rules
    )

    trainer_rules.train()

    _ = trainer_rules.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True


def test_cleanlab_base_test_signatures(std_trainer_input_2):
    (
        model,
        inputs_x,
        mapping_rules_labels_t,
        train_rule_matches_z,
        test_dataset,
        test_labels
    ) = std_trainer_input_2

    config_signatures = CleanLabConfig(
        cv_n_folds=2, criterion=CrossEntropyLoss, psx_calculation_method="signatures", seed=1234
    )

    trainer_signatures = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config_signatures
    )
    trainer_signatures.train()

    _ = trainer_signatures.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True
