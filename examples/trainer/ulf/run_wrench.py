import argparse
import json
import logging
import os
import random
import sys
from itertools import product

import torch

from examples.trainer.ulf.ulf_training_bert_cosine import load_model
from wrench._logging import LoggingHandler
from wrench.dataset import load_dataset
from wrench.endmodel import Cosine, EndClassifierModel
from snorkel.utils import probs_to_preds

#### Just some code to print debug information to stdout
from wrench.evaluation import METRIC
from wrench.labelmodel import MajorityVoting, Snorkel, FlyingSquid

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

num_parameter_settings = 10  # how many random parameter combinations will be tried
n_steps = 100000
batch_size = 8
test_batch_size = 8
patience = 200
evaluation_step = 50


def return_random_combinations(parameters):
    # select some number of parameter combination that we will try
    parameter_values = [v for v in parameters.values()]
    parameter_combinations = []
    for run_id, (params) in enumerate(product(*parameter_values)):
        parameter_combinations.append(params)
    return random.choices(parameter_combinations, k=num_parameter_settings)


def train_wrench_mv_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_model = MajorityVoting(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    soft_label = label_model.predict_proba(train_data)
    hard_label = probs_to_preds(soft_label)

    all_metrics = []
    parameters = dict(
        lr=[2e-5, 3e-5, 5e-5],
        # batch_size=[8]      #16, 32
    )
    random_combinations = return_random_combinations(parameters)

    for params in random_combinations:
        lr = params[0]
        # params_dict = {"lr": rob_lr, "batch_size": rob_batch_size}
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        model = EndClassifierModel(
            batch_size=batch_size, test_batch_size=test_batch_size, n_steps=n_steps,
            optimizer='AdamW', optimizer_lr=lr, backbone='BERT', backbone_model_name=model_name,
        )
        model.fit(
            dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label,
            device=device, metric=target, patience=patience, evaluation_step=evaluation_step,
        )

        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_mv_r_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_snorkel_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parameters = dict(
        dp_lr=[1e-5, 5e-5, 1e-4],
        dp_weight_decay=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        dp_num_epochs=[5, 10, 50, 100, 200],
        lr=[2e-5, 3e-5, 5e-5],  # 0.00001,
        # batch_size=[8]      # 16, 32
    )
    random_combinations = return_random_combinations(parameters)

    all_metrics = []
    for params in random_combinations:
        dp_lr, dp_weight_decay, dp_num_epochs, lr = params
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        print(f"Parameters: {str(params_dict)}")

        label_model = Snorkel(
            lr=dp_lr, l2=0.0, n_epochs=dp_num_epochs, weight_decay=dp_weight_decay, batch_size=batch_size,
            test_batch_size=test_batch_size
        )
        label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
        # train_data = train_data.get_covered_subset()
        aggregated_hard_labels = label_model.predict(train_data)

        # Run end model: BERT
        model = EndClassifierModel(
            batch_size=batch_size, test_batch_size=test_batch_size, n_steps=n_steps,
            optimizer='AdamW', optimizer_lr=lr, backbone='BERT', backbone_model_name=model_name,
        )
        model.fit(
            dataset_train=train_data, dataset_valid=valid_data, y_train=aggregated_hard_labels,
            device=device, metric=target, patience=patience, evaluation_step=evaluation_step,
        )
        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_snorkel_r_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_mv_cosine_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_model = MajorityVoting(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    soft_label = label_model.predict_proba(train_data)
    hard_label = probs_to_preds(soft_label)

    parameters = dict(
        # batch_size=[8],         # 32
        lr=[1e-6, 1e-5],
        decay=[1e-4],
        update=[50, 100, 200],
        thresh=[0.2, 0.4, 0.6, 0.8],
        lam=[0.01, 0.05, 0.1],  # 0.05
        mu=[1.0],
        margin=[1.0]
    )
    random_combinations = return_random_combinations(parameters)

    all_metrics = []
    for params in random_combinations:
        lr, decay, update, thresh, lam, mu, margin = params
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        print(f"Parameters: {str(params_dict)}")

        model = Cosine(
            n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size,
            optimizer="AdamW", optimizer_lr=lr, optimizer_weight_decay=decay,
            teacher_update=update, thresh=thresh, lamda=lam, mu=mu, margin=margin,
            backbone='BERT', backbone_model_name=model_name
        )
        model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label, device=device,
                  metric=target, patience=patience, evaluation_step=evaluation_step)

        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_mv_rc_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_snorkel_cosine_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parameters = dict(
        dp_lr=[1e-5, 5e-5, 1e-4],
        dp_weight_decay=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        dp_num_epochs=[5, 10, 50, 100, 200],
        # batch_size=[8],
        lr=[1e-6, 1e-5],
        decay=[1e-4],
        update=[50, 100, 200],
        thresh=[0.2, 0.4, 0.6, 0.8],
        lam=[0.01, 0.05, 0.1],  # 0.05
        mu=[1.0],
        margin=[1.0]
    )
    random_combinations = return_random_combinations(parameters)

    all_metrics = []
    for params in random_combinations:
        dp_lr, dp_weight_decay, dp_num_epochs, lr, decay, update, thresh, lam, mu, margin = params
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        print(f"Parameters: {str(params_dict)}")

        label_model = Snorkel(lr=dp_lr, l2=0.0, n_epochs=dp_num_epochs, weight_decay=dp_weight_decay,
                              batch_size=batch_size, test_batch_size=test_batch_size)
        label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
        aggregated_hard_labels = label_model.predict(train_data)

        model = Cosine(
            n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size,
            optimizer="AdamW", optimizer_lr=lr, optimizer_weight_decay=decay,
            teacher_update=update, thresh=thresh, lamda=lam, mu=mu, margin=margin,
            backbone='BERT', backbone_model_name=model_name
        )
        model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=aggregated_hard_labels, device=device,
                  metric=target, patience=patience, evaluation_step=evaluation_step)

        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_snorkel_rc_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_gold(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    lrs = [2e-5, 3e-5, 5e-5]
    all_metrics = []

    # Run end model: BERT
    for lr in lrs:
        params_dict = {"lr": lr}
        model = EndClassifierModel(
            batch_size=batch_size, test_batch_size=test_batch_size, n_steps=n_steps,
            optimizer='AdamW', optimizer_lr=lr,
            backbone='BERT', backbone_model_name=model_name,
        )
        model.fit(
            dataset_train=train_data, dataset_valid=valid_data, y_train=train_data.labels,
            device=device, metric=target, patience=patience, evaluation_step=evaluation_step,
        )
        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_supervised_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_fs_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_model = FlyingSquid()
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    soft_label = label_model.predict_proba(train_data)
    hard_label = probs_to_preds(soft_label)

    all_metrics = []
    parameters = dict(
        lr=[2e-5, 3e-5, 5e-5],
        # batch_size=[8]          #16, 32
    )
    random_combinations = return_random_combinations(parameters)

    for params in random_combinations:
        lr = params[0]
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        model = EndClassifierModel(
            batch_size=batch_size, test_batch_size=test_batch_size, n_steps=n_steps,
            optimizer='AdamW', optimizer_lr=lr,
            backbone='BERT', backbone_model_name=model_name,
        )
        model.fit(
            dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label,
            device=device, metric=target, patience=patience, evaluation_step=evaluation_step,
        )
        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_fs_r_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


def train_wrench_fs_cosine_roberta(path: str, dataset: str, output_path: str, target: str, model):
    output_path = os.path.join(output_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    model_name = load_model(model)
    train_data, valid_data, test_data = load_dataset(path, dataset, dataset_type="TextDataset", extract_feature=True,
                                                     extract_fn="bert", cache_name=model, model_name=model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_model = FlyingSquid()
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    soft_label = label_model.predict_proba(train_data)
    hard_label = probs_to_preds(soft_label)

    parameters = dict(
        lr=[1e-6, 1e-5],
        decay=[1e-4],
        update=[50, 100, 200],
        thresh=[0.2, 0.4, 0.6, 0.8],
        lam=[0.01, 0.05, 0.1],  # 0.05
        mu=[1.0],
        margin=[1.0]
    )
    random_combinations = return_random_combinations(parameters)

    all_metrics = []
    for params in random_combinations:
        lr, decay, update, thresh, lam, mu, margin = params
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        print(f"Parameters: {str(params_dict)}")

        model = Cosine(
            n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size,
            optimizer="AdamW", optimizer_lr=lr, optimizer_weight_decay=decay,
            teacher_update=update, thresh=thresh, lamda=lam, mu=mu, margin=margin,
            backbone='BERT', backbone_model_name=model_name
        )
        model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label, device=device,
                  metric=target, patience=patience, evaluation_step=evaluation_step)

        curr_metric_valid = model.test(valid_data, target)
        curr_metric_test = model.test(test_data, target)
        metrics = {f"Valid {target}": curr_metric_valid, f"Test {target}": curr_metric_test}

        print(f"Valid {target}: {curr_metric_valid}")
        print(f"Test {target}: {curr_metric_test}")

        all_metrics.append({**params_dict, **metrics})

    with open(os.path.join(output_path, f"wrench_{dataset}_fs_rc_results.json"), 'a+') as file:
        json.dump(all_metrics, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--output_path", help="")
    parser.add_argument("--model", help="")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_gold(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "youtube", args.output_path, "acc", args.model)
    #
    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_gold(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "sms", args.output_path, "f1_binary", args.model)
    #
    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    # ### no gold data for spouse -> no training
    # train_wrench_mv_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "spouse", args.output_path, "f1_binary", args.model)
    #
    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_gold(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "trec", args.output_path, "acc", args.model)
    #
    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_gold(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "yelp", args.output_path, "acc", args.model)
    #
    # logger.info("New dataset!")
    # #
    # train_wrench_fs_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_gold(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "semeval", args.output_path, "acc", args.model)

    # logger.info("New dataset!")
    #
    # train_wrench_fs_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_fs_cosine_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_gold(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_mv_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_snorkel_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_mv_cosine_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    # train_wrench_snorkel_cosine_roberta(args.path_to_data, "agnews", args.output_path, "acc", args.model)
    #
    logger.info("New dataset!")

    train_wrench_fs_roberta(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")
    train_wrench_gold(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")
    train_wrench_mv_roberta(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")
    train_wrench_snorkel_roberta(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")
    train_wrench_mv_cosine_roberta(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")
    train_wrench_snorkel_cosine_roberta(args.path_to_data, "yoruba", args.output_path, "f1_macro", "multilingual")

    logger.info("New dataset!")

    train_wrench_fs_roberta(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
    train_wrench_gold(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
    train_wrench_mv_roberta(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
    train_wrench_snorkel_roberta(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
    train_wrench_mv_cosine_roberta(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
    train_wrench_snorkel_cosine_roberta(args.path_to_data, "hausa", args.output_path, "f1_macro", "multilingual")
