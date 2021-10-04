from torchdata.datasets import TensorDataset


@pytest.fixture
def cleanlab_input():
    num_samples = 64
    num_features = 16
    num_rules = 6
    num_classes = 2

    x_np = np.ones((num_samples, num_features)).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)
    model_input_x = TensorDataset(x_tensor)

    rule_matches_z = np.zeros((num_samples, num_rules))
    rule_matches_z[0, 0] = 1
    rule_matches_z[1:, 1] = 1

    mapping_rules_labels_t = np.zeros((num_rules, num_classes))
    mapping_rules_labels_t[:, 1] = 1

    y_np = np.ones((num_samples,))
    y_labels = TensorDataset(torch.from_numpy(y_np))

    model = LogisticRegressionModel(num_features, num_classes)

    return (
        model,
        model_input_x, rule_matches_z, mapping_rules_labels_t,
        y_labels
    )