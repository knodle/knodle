from typing import Dict

import pandas as pd
import numpy as np

from knodle.transformation.majority import z_t_matrices_to_majority_vote_labels


def get_y_statistics(y_gold: np.ndarray) -> pd.DataFrame:
    y = pd.Series(y_gold)
    stats_dict = [
        ["num_classes", y.nunique()],
        ["num_samples", y_gold.shape[0]],
        ["classes", y.unique().tolist()],
        ["class_sums", pd.Series(y_gold).value_counts().to_dict()],
        ["class_distribution", (pd.Series(y_gold).value_counts() / y_gold.shape[0]).to_dict()]
    ]
    stats_dict = pd.DataFrame(stats_dict, columns=["statistic", "value"])
    return stats_dict


def get_z_statistics(rule_matches_z: np.array, mapping_rules_labels_t: np.array) -> Dict:
    hits = np.round(rule_matches_z.sum(axis=0) / rule_matches_z.shape[0], 2)
    no_match_samples = np.where(hits == 0)[0].sum() / rule_matches_z.sum()

    avg_hits_per_sample = rule_matches_z.sum() / rule_matches_z.shape[0]

    majority_votes = z_t_matrices_to_majority_vote_labels(
        rule_matches_z, mapping_rules_labels_t, choose_random_label=False, other_class_id=-1
    )
    majority_votes = pd.Series(majority_votes)

    stats_dict = {
        "num_samples": rule_matches_z.shape[0],
        "num_classes": mapping_rules_labels_t.shape[1],
        "num_weak_labels": rule_matches_z.shape[1],
        "samples_without_match": no_match_samples,
        "avg_hits_per_sample": avg_hits_per_sample
    }

    return stats_dict


def get_standard_paper_stats(rule_matches_z: np.array, mapping_rules_labels_t: np.array, y_gold: np.ndarray):
    y = pd.Series(y_gold)

    if mapping_rules_labels_t.shape[1] == 2:
        skewdness = round(y_gold.sum() / y_gold.shape[0], 2)
    else:
        skewdness = None

    stats_dict = [
        ["classes", (y_gold.max() + 1).astype(str)],
        ["train / test samples", f"{rule_matches_z.shape[0]} / {y_gold.shape[0]}"],
        ["rules", rule_matches_z.shape[1]],
        ["avg. rule hits", round(rule_matches_z.sum() / rule_matches_z.shape[0], 2)],
        ["skewdness", skewdness]
    ]
    stats_dict = pd.DataFrame(stats_dict, columns=["statistic", "value"])
    return stats_dict


def combine_multiple_paper_stats(dataset_to_df_dict: Dict) -> pd.DataFrame:

    columns = []
    values = []

    for dataset, df in dataset_to_df_dict.items():
        columns = df["statistic"].tolist()
        values.append([dataset] + df["value"].tolist())

    stats_df = pd.DataFrame(values, columns=["dataset"] + columns)
    return stats_df