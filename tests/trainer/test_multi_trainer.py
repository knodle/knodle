from tests.trainer.generic import std_trainer_input_1

from knodle.trainer.multi_trainer import MultiTrainer


def test_auto_train(std_trainer_input_1):
    (
        model,
        model_input_x, rule_matches_z, mapping_rules_labels_t,
        y_labels
    ) = std_trainer_input_1

    trainers = ["majority", "snorkel", "knn", "snorkel_knn"]
    trainer = MultiTrainer(
        name=trainers,
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=model_input_x,
        rule_matches_z=rule_matches_z,
    )

    trainer.train()
    metrics = trainer.test(model_input_x, y_labels)

    # Check whether the code ran up to here
    assert 2 == 2
