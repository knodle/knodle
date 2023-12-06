import argparse
import json
import logging
import os
import random
import sys
from itertools import product

import pandas as pd
import torch
from joblib import load
from torch.nn import CrossEntropyLoss
from transformers import AdamW

from examples.trainer.ulf.utils import WrenchDataset, z_to_wrench_labels, bert_text_extractor
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.deep_ulf import UlfTrainer
from wrench.endmodel import Cosine
from wrench.utils import set_seed

logger = logging.getLogger(__name__)


def train_ulf_bert(path: str, dataset: str, output_path: str, target: str, model: str = "roberta") -> None:
    """ This is an example of launching cleanlab trainer with BERT model """

    seed = 12345
    set_seed(seed)

    assert target in ['acc', 'f1_binary', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path_to_data, label_path = load_paths(path, dataset)
    model_name = load_model(model)

    # load data from files
    df_train, df_dev, df_test, train_z, dev_z, test_z, t = load_data(path_to_data)

    # convert input data to features
    train_features, dev_features, test_features = load_features(df_train, df_dev, df_test, model, path_to_data, model_name, device)

    # calculate weak labels
    train_weak_labels, dev_weak_labels, test_weak_labels = get_weak_labels(train_z, dev_z, test_z, t)

    # gold labels for dev and test sets
    dev_labels = df_dev["label"].tolist()
    test_labels = df_test["label"].tolist()

    n_classes = max(test_labels) + 1
    n_lf = t.shape[0]
    id2label = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}

    # transform data to wrench datasets
    train_data = WrenchDataset(features=train_features, n_class=n_classes, n_lf=n_lf,
                               examples=[{"text": sample} for sample in list(df_train["sample"])],
                               weak_labels=train_weak_labels, id2label=id2label)
    dev_data = WrenchDataset(features=dev_features, n_class=n_classes, n_lf=n_lf, labels=dev_labels,
                             examples=[{"text": sample} for sample in list(df_dev["sample"])],
                             weak_labels=dev_weak_labels, id2label=id2label)
    test_data = WrenchDataset(features=test_features, n_class=n_classes, n_lf=n_lf, labels=test_labels,
                              examples=[{"text": sample} for sample in list(df_test["sample"])],
                              weak_labels=test_weak_labels, id2label=id2label)

    num_experiments = 1
    num_parameter_settings = 10

    parameters = dict(
        p=[0.1, 0.3, 0.5, 0.7, 0.9],
        lr=[1e-4, 1e-5, 1e-6],  # 0.00001,

        folds=[2, 3, 5, 7],        # , 7
        iterations=[4],
        partitions=[1],

        lamda=[0.01, 0.05, 0.1],  # 0.05
        thresh=[0.2, 0.4, 0.6, 0.8],
        update=[50, 100, 200],
        margin=[1.0]
    )

    parameter_values = [v for v in parameters.values()]

    n_steps = 100000
    batch_size = 8
    test_batch_size = 8
    patience = 200
    evaluation_step = 50
    other_coeff = 0  # do not include non-labeled samples to the training folds
    optimizer_weight_decay = 1e-4
    mu = 1.0

    # select some number of parameter combination that we will try
    parameter_combinations = []
    for run_id, (params) in enumerate(product(*parameter_values)):
        parameter_combinations.append(params)
    random_combinations = random.choices(parameter_combinations, k=num_parameter_settings)  # how many random parameter combinations will be tried

    all_results = []

    for params_num, params in enumerate(random_combinations):
        p, lr, folds, iterations, partitions, lamda, thresh, update, margin = params
        params_dict = {k: v for k, v in locals().items() if (k in list(parameters.keys())) and (not k.startswith('__'))}
        print(str(params_dict))

        for exp in range(0, num_experiments):
            print(f"Experiment: {exp + 1} out of {num_experiments}")
            model = Cosine(
                n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size,
                optimizer="AdamW",
                optimizer_lr=lr, optimizer_weight_decay=optimizer_weight_decay, thresh=thresh, lamda=lamda, mu=mu,
                backbone='BERT', backbone_model_name=model_name,
                teacher_update=update, margin=margin,
            )
            custom_cleanlab_config = UlfConfig(
                iterations=iterations,
                cv_n_folds=folds,
                partitions=partitions,
                psx_calculation_method="signatures",
                use_probabilistic_labels=False,
                use_prior=False,
                p=p,
                output_classes=n_classes,
                criterion=CrossEntropyLoss,
                epochs=20,
                grad_clipping=5,
                evaluation_step=evaluation_step,
                patience=patience,
                save_model_path=output_path,
                save_model_name=f'{dataset}_{"_".join([f"{key}-{value}" for key, value in params_dict.items()])}_{params_num}_exp{exp}',
                optimizer=AdamW,
                lr=lr,
                batch_size=64,
                early_stopping=True,
                other_coeff=other_coeff,

                psx_epochs=20,
                psx_lr=0.8,
                psx_optimizer=AdamW,

                device=device,
                target=target
            )
            logger.info(custom_cleanlab_config)
            trainer = UlfTrainer(
                model=model,
                mapping_rules_labels_t=t,
                model_input_x=train_data,
                rule_matches_z=train_z,
                trainer_config=custom_cleanlab_config,

                dev_model_input_x=dev_data,        # NB! Here it is not just X data, but it is the whole wrench dataset
                dev_gold_labels_y=dev_labels
            )

            results = trainer.train(test_data=test_data, target=target)

            print(f"Experiment {exp + 1} is done")
            print(f"Parameters: {str(params_dict)}")
            print(f"Results: {results}")

            all_results.append({**params_dict, **results})
            with open(os.path.join(output_path, f"{dataset}_final_results.json"), 'w') as file:
                json.dump(all_results, file)


            # acc.append(float(trainer.model.test(test_data, "acc")))
            # f1_macro.append(float(trainer.model.test(test_data, "f1_macro")))
            # f1_micro.append(float(trainer.model.test(test_data, "f1_micro")))

        # log_section(str(params_dict), logger)
        # log_section(f"Accuracy: {' '.join(map(str, acc))}", logger)
        # log_section(f"F1_macro: {' '.join(map(str, f1_macro))}", logger)
        # log_section(f"F1_micro: {' '.join(map(str, f1_micro))}", logger)

        # res = collect_res_statistics(acc=acc, f1_macro=f1_macro, f1_micro=f1_micro, params=params_dict)
        # with open(os.path.join(output_path, f"{dataset}_final_results.json"), 'a+') as file:
        #     json.dump(res, file)

        # if results is not None and len(results) == 3:
        #     accs_iters.append(results[0])
        #     f1_macros_iters.append(results[1])
        #     f1_micros_iters.append(results[2])
        #
        #     for i in range(iterations):
        #         curr_params_dict = copy.deepcopy(params_dict)
        #         curr_params_dict["iterations"] = i + 1
        #         curr_params_dict["label_types"] = "hard" if i % 2 == 0 or i == 0 else "soft"
        #         res = collect_res_statistics(
        #             acc=list(list(zip(*accs_iters))[i]),
        #             f1_macro=list(list(zip(*f1_macros_iters))[i]),
        #             f1_micro=list(list(zip(*f1_micros_iters))[i]),
        #             params=curr_params_dict, verbose=False
        #         )
        #         print(res)
        #         with open(os.path.join(output_path, f"{dataset}_final_results.json"), 'a+') as file:
        #             json.dump(res, file)

def load_data(path_to_data):
    df_train = pd.read_csv(os.path.join(path_to_data, 'train_df.csv'), sep="\t")
    df_dev = pd.read_csv(os.path.join(path_to_data, 'dev_df.csv'), sep="\t")
    df_test = pd.read_csv(os.path.join(path_to_data, 'test_df.csv'), sep="\t")
    train_z = load(os.path.join(path_to_data, 'train_rule_matches_z.lib'))
    dev_z = load(os.path.join(path_to_data, 'dev_rule_matches_z.lib'))
    test_z = load(os.path.join(path_to_data, 'test_rule_matches_z.lib'))
    t = load(os.path.join(path_to_data, 'mapping_rules_labels_t.lib'))
    return df_train, df_dev, df_test, train_z, dev_z, test_z, t


def load_paths(path, dataset):
    return os.path.join(path, dataset), os.path.join(path, dataset, 'label.json')


def get_weak_labels(train_z, dev_z, test_z, t):
    # get weak labels in wrench/snorkel format
    train_weak_labels = z_to_wrench_labels(train_z, t)
    dev_weak_labels = z_to_wrench_labels(dev_z, t)
    test_weak_labels = z_to_wrench_labels(test_z, t)
    return train_weak_labels, dev_weak_labels, test_weak_labels


def load_features(df_train, df_dev, df_test, model, path_to_data, model_name, device):
    # get bert features
    train_features = bert_text_extractor(data=list(df_train["sample"]), path=path_to_data, split="train",
                                         cache_name=model, model_name=model_name, device=device)
    dev_features = bert_text_extractor(data=list(df_dev["sample"]), path=path_to_data, split="dev",
                                       cache_name=model, model_name=model_name, device=device)
    test_features = bert_text_extractor(data=list(df_test["sample"]), path=path_to_data, split="test",
                                        cache_name=model, model_name=model_name, device=device)
    return train_features, dev_features, test_features


def load_model(model: str):
    if model == "roberta":
        return "roberta-base"
    elif model == "bert":
        return "bert-base-cased"
    elif model == "multilingual":
        return "bert-base-multilingual-cased"
    else:
        raise ValueError("Check the model name")


def get_param_dict(params, target):
    p, lr, folds, iterations, partitions, lamda, thresh = params
    param_dict = {"p": p, "lr": lr, "folds": folds, "iter": iterations,
                  "part": partitions, "l": lamda, "thres": thresh, "target": target}
    print(str(param_dict))
    return param_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_to_data", help="")
    parser.add_argument("--dataset", help="")
    parser.add_argument("--output_path", help="")
    parser.add_argument("--target", help="")
    parser.add_argument("--model", help="", default="roberta")

    args = parser.parse_args()

    output_path = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    if args.dataset in ["yoruba", "hausa"] and args.model != "multilingual":
        print("African datasets can be encoded only with multilingual BERT. The model is changed to 'multilingual'")
        model = "multilingual"
    elif args.dataset not in ["yoruba", "hausa"] and args.model == "multilingual":
        print("Why do you want to use multilingual BERT if your datasets English? The model is changed to 'roberta'")
        model = "roberta"
    else:
        model = args.model

    train_ulf_bert(args.path_to_data, args.dataset, output_path, args.target, model)
