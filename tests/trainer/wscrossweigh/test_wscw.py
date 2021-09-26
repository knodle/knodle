import shutil
from torch.nn import CrossEntropyLoss

from knodle.trainer import WSCrossWeighConfig, WSCrossWeighTrainer
from tests.trainer.generic import std_trainer_input_2


def test_dscw_base_test(std_trainer_input_2):
    (
        model,
        inputs_x, mapping_rules_labels_t, train_rule_matches_z,
        test_dataset, test_labels
    ) = std_trainer_input_2

    config = WSCrossWeighConfig(folds=2)

    trainer = WSCrossWeighTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)

    shutil.rmtree(trainer.trainer_config.caching_folder)

    # Check that this runs without error
    assert True


def test_dscw_base_test_with_CE_loss(std_trainer_input_2):
    (
        model,
        inputs_x, mapping_rules_labels_t, train_rule_matches_z,
        test_dataset, test_labels
    ) = std_trainer_input_2

    config = WSCrossWeighConfig(folds=2, criterion=CrossEntropyLoss, use_probabilistic_labels=False)

    trainer = WSCrossWeighTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)

    shutil.rmtree(trainer.trainer_config.caching_folder)

    # Check that this runs without error
    assert True
