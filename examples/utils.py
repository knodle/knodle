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
        exp_results_acc: List, exp_results_prec: List, exp_results_recall: List, exp_results_f1_avg: List,
        exp_results_f1_weighted: List, params: Dict = None, verbose: bool = False
) -> Dict:
    num_experiments = len(exp_results_acc)
    result = {
        "accuracy": exp_results_acc,
        "mean_accuracy": statistics.mean(exp_results_acc),
        "std_accuracy": statistics.stdev(exp_results_acc),
        "sem_accuracy": sem(exp_results_acc),
        #
        "precision": exp_results_prec,
        "mean_precision": statistics.mean(exp_results_prec),
        "std_precision": statistics.stdev(exp_results_prec),
        "sem_precision": sem(exp_results_prec),
        #
        "recall": exp_results_recall,
        "mean_recall": statistics.mean(exp_results_recall),
        "std_recall": statistics.stdev(exp_results_recall),
        "sem_recall": sem(exp_results_recall),
        #
        "f1_avg": exp_results_f1_avg,
        "mean_f1_avg": statistics.mean(exp_results_f1_avg),
        "std_f1_avg": statistics.stdev(exp_results_f1_avg),
        "sem_f1_avg": sem(exp_results_f1_avg),
        #
        "f1_weighted": exp_results_f1_weighted,
        "mean_f1_weighted": statistics.mean(exp_results_f1_weighted),
        "std_f1_weighted": statistics.stdev(exp_results_f1_weighted),
        "sem_f1_weighted": sem(exp_results_f1_weighted),
    }
    if params:
        result = {**params, **result}

    if verbose:
        log_section(
            f"Experiments: {num_experiments} \n"
            f"Average accuracy: {result['mean_accuracy']}, std: {result['std_accuracy']}, "
            f"sem: {result['sem_accuracy']} \n "
            f"Average prec: {result['mean_precision']}, std: {result['std_precision']}, "
            f"sem: {result['sem_precision']} \n"
            f"Average recall: {result['mean_recall']}, std: {result['std_recall']}, "
            f"sem: {result['sem_recall']} \n"
            f"Average F1 Avg: {result['mean_f1_avg']}, std: {result['std_f1_avg']}, "
            f"sem: {result['sem_f1_avg']} \n"
            f"Average F1 Weighted: {result['mean_f1_weighted']}, std: {result['std_f1_weighted']}, "
            f"sem: {result['sem_f1_weighted']}",
            logger
        )
    return result


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