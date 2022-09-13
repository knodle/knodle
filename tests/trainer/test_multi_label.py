from torch.nn import BCEWithLogitsLoss

from knodle.trainer import MajorityVoteTrainer, MajorityConfig


def test_auto_train(std_trainer_input_1):
    (
        model,
        model_input_x, rule_matches_z, mapping_rules_labels_t,
        _
    ) = std_trainer_input_1

    y_labels = [[1, 0]] * 25 + [[1]] * 25 + [[0]] * 14
    curr_config = MajorityConfig(multi_label=True, multi_label_threshold=0.7, criterion=BCEWithLogitsLoss)

    trainer = MajorityVoteTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=model_input_x,
        rule_matches_z=rule_matches_z,
        trainer_config=curr_config
    )

    trainer.train()
    metrics, _ = trainer.test(model_input_x, y_labels)

    # Check whether the code ran up to here
    assert True
