from tests.trainer.generic import std_trainer_input_2

from knodle.trainer.wscrossweigh.wscrossweigh_weights_calculator import WSCrossWeighWeightsCalculator


def test_dscw_base_test(std_trainer_input_2):
    (
        model,
        inputs_x, mapping_rules_labels_t, train_rule_matches_z,
        test_dataset, test_labels
    ) = std_trainer_input_2

    trainer = WSCrossWeighWeightsCalculator(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True
