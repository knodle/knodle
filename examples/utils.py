import logging
import os
import statistics
from typing import Union, List, Tuple, Dict

import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import sem

from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)


def read_train_dev_test(
        target_path: str, if_dev_data: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
           Tuple[pd.DataFrame, None, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """This function loads the matrices as well as train, test (occasionally, also dev) data from corresponding files"""
    df_train = load(os.path.join(target_path, 'df_train.lib'))
    df_test = load(os.path.join(target_path, 'df_test.lib'))
    z_train_rule_matches = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
    z_test_rule_matches = load(os.path.join(target_path, 'test_rule_matches_z.lib'))
    t_mapping_rules_labels = load(os.path.join(target_path, 'mapping_rules_labels_t.lib'))

    if if_dev_data:
        dev_df = load(os.path.join(target_path, 'df_dev.lib'))
        return df_train, dev_df, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels

    return df_train, None, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels


def get_samples_list(data: Union[pd.Series, pd.DataFrame], column_num: int = None) -> List:
    """ Extracts the data from the Series/DataFrame and returns it as a list"""
    column_num = int(column_num)
    if isinstance(data, pd.Series):
        return list(data)
    elif isinstance(data, pd.DataFrame) and column_num is not None:
        return list(data.iloc[:, column_num])
    else:
        raise ValueError(
            "Please pass input data either as a Series or as a DataFrame with number of the column with samples"
        )


def collect_res_statistics(
        acc: List = None, prec: List = None, recall: List = None, f1_macro: List = None, f1_micro: List = None,
        f1_avg: List = None, f1_weighted: List = None, f1_binary: List = None, params: Dict = None, verbose: bool = False
) -> Dict:
    num_experiments = len(acc)
    results = {}

    if acc is not None:
        results["accuracy"] = acc
        if len(acc) > 1:
            results["mean_accuracy"] = statistics.mean(acc)
            results["std_accuracy"] = statistics.stdev(acc)
            results["sem_accuracy"] = sem(acc)

    if prec is not None:
        results["precision"] = prec
        if len(prec) > 1:
            results["mean_precision"] = statistics.mean(prec)
            results["std_precision"] = statistics.stdev(prec)
            results["sem_precision"] = sem(prec)

    if recall is not None:
        results["recall"] = recall
        if len(recall) > 1:
            results["mean_recall"] = statistics.mean(recall)
            results["std_recall"] = statistics.stdev(recall)
            results["sem_recall"] = sem(recall)

    if f1_macro is not None:
        results["f1_macro"] = f1_macro
        if len(f1_macro) > 1:
            results["mean_f1_macro"] = statistics.mean(f1_macro)
            results["std_f1_macro"] = statistics.stdev(f1_macro)
            results["sem_f1_macrog"] = sem(f1_macro)

    if f1_micro is not None:
        results["f1_micro"] = f1_micro
        if len(f1_micro) > 1:
            results["mean_f1_micro"] = statistics.mean(f1_micro)
            results["std_f1_micro"] = statistics.stdev(f1_micro)
            results["sem_f1_micro"] = sem(f1_micro)

    if f1_avg is not None:
        results["f1_avg"] = f1_avg
        if len(f1_avg) > 1:
            results["mean_f1_avg"] = statistics.mean(f1_avg)
            results["std_f1_avg"] = statistics.stdev(f1_avg)
            results["sem_f1_avg"] = sem(f1_avg)

    if f1_weighted is not None:
        results["f1_weighted"] = f1_weighted
        if len(f1_weighted) > 1:
            results["mean_f1_weighted"] = statistics.mean(f1_weighted)
            results["std_f1_weighted"] = statistics.stdev(f1_weighted)
            results["sem_f1_weighted"] = sem(f1_weighted)

    if f1_binary is not None:
        results["f1_binary"] = f1_weighted
        if len(f1_weighted) > 1:
            results["mean_f1_binary"] = statistics.mean(f1_weighted)
            results["std_f1_binary"] = statistics.stdev(f1_weighted)
            results["sem_f1_binary"] = sem(f1_weighted)

    if params:
        results = {**params, **results}

    if verbose:
        log_section(
            f"Experiments: {num_experiments} \n"
            f"Average accuracy: {results['mean_accuracy']}, std: {results['std_accuracy']}, "
            f"sem: {results['sem_accuracy']} \n "
            f"Average prec: {results['mean_precision']}, std: {results['std_precision']}, "
            f"sem: {results['sem_precision']} \n"
            f"Average recall: {results['mean_recall']}, std: {results['std_recall']}, "
            f"sem: {results['sem_recall']} \n"
            f"Average F1 Avg: {results['mean_f1_avg']}, std: {results['std_f1_avg']}, "
            f"sem: {results['sem_f1_avg']} \n"
            f"Average F1 Weighted: {results['mean_f1_weighted']}, std: {results['std_f1_weighted']}, "
            f"sem: {results['sem_f1_weighted']}",
            logger
        )
    return results


def print_metrics(clf_report):
    logger.info(f"Accuracy is: {clf_report['accuracy']}")
    logger.info(f"Precision is: {clf_report['macro avg']['precision']}")
    logger.info(f"Recall is: {clf_report['macro avg']['recall']}")
    logger.info(f"Avg F1 is: {clf_report['macro avg']['f1-score']}")
    logger.info(f"Weighted F1 is: {clf_report['weighted avg']['f1-score']}")


"""

[{
    'seed': None, 'caching_suffix': '', 'caching_folder': '/content/cache', 'saved_models_dir': '/content/cache', 
    'criterion': <function cross_entropy_with_probs at 0x7fccc3dc5950>, 'lr': 0.0001, 'batch_size': 16, 
    'output_classes': 2, 'grad_clipping': None, 'device': device(type='cuda'), 
    'epochs': 3, 'optimizer': <class 'torch.optim.adamw.AdamW'>, 'class_weights': tensor([1., 1.]), 
    'filter_non_labelled': True, 'other_class_id': None, 'evaluate_with_other_class': False, 'ids2labels': None, 
    'max_rules': None, 'min_coverage': None, 'drop_rules': False, 'use_probabilistic_labels': True, 
    'probability_threshold': None
}, 
{
    'seed': 12345, 'caching_suffix': '', 'caching_folder': '/content/cache', 'saved_models_dir': '/content/cache', 
    'criterion': <function cross_entropy_with_probs at 0x7fccc3dc5950>, 'lr': 0.0001, 'batch_size': 32, 
    'output_classes': 2, 'grad_clipping': None, 'device': device(type='cuda'), 'epochs': 2, 
    'optimizer': <class 'torch.optim.adamw.AdamW'>, 'class_weights': tensor([1., 1.]), 'filter_non_labelled': True, 
    'other_class_id': None, 'evaluate_with_other_class': False, 'ids2labels': None, 'max_rules': None, 
    'min_coverage': None, 'drop_rules': False, 'use_probabilistic_labels': True, 'probability_threshold': None, 'k': 2, 
    'radius': None, 'use_approximation': False, 'activate_no_match_instances': True, 'n_jobs': 4
}, 
{   
    'seed': None, 'caching_suffix': '', 'caching_folder': '/content/cache', 'saved_models_dir': '/content/cache', 
    'criterion': <function cross_entropy_with_probs at 0x7fccc3dc5950>, 'lr': 0.01, 'batch_size': 32, 
    'output_classes': 2, 'grad_clipping': None, 'device': device(type='cuda'), 'epochs': 3, 
    'optimizer': <class 'torch.optim.adamw.AdamW'>, 'class_weights': tensor([1., 1.]), 
    'filter_non_labelled': True, 'other_class_id': None, 'evaluate_with_other_class': False, 'ids2labels': None, 
    'max_rules': None, 'min_coverage': None, 'drop_rules': False, 'use_probabilistic_labels': True, 
    'probability_threshold': None, 'label_model_num_epochs': 5000, 'label_model_log_freq': 500
}, 
{
    'seed': 111, 'caching_suffix': '', 'caching_folder': '/content/cache', 'saved_models_dir': '/content/cache', 
    'criterion': <function cross_entropy_with_probs at 0x7fccc3dc5950>, 'lr': 0.01, 'batch_size': 32, 
    'output_classes': 2, 'grad_clipping': None, 'device': device(type='cuda'), 'epochs': 3, 
    'optimizer': <class 'torch.optim.adamw.AdamW'>, 'class_weights': tensor([1., 1.]), 'filter_non_labelled': True, 
    'other_class_id': None, 'evaluate_with_other_class': False, 'ids2labels': None, 'max_rules': None, 
    'min_coverage': None, 'drop_rules': False, 'use_probabilistic_labels': True, 'probability_threshold': None, 
    'draw_plot': False, 'partitions': 2, 'folds': 10, 'weight_reducing_rate': 0.5, 'samples_start_weights': 2.0, 
    'cw_grad_clipping': None, 'cw_epochs': 3, 'cw_batch_size': 32, 'cw_optimizer': <class 'torch.optim.adamw.AdamW'>, 
    'cw_filter_non_labelled': True, 'cw_other_class_id': None, 'cw_seed': None, 'cw_lr': 0.1
}]

"""