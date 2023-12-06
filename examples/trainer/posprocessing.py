import os
import json
import csv
from typing import List


def postprocess_json(input_file: str, required_metrics: List, output_file: str) -> None:

    with open(input_file, encoding="utf-8") as file:
        results = json.load(file)

    # header = ['psx', 'prior', 'p', 'lr', 'folds', 'iterations']
    header = ['psx', 'lr', 'folds', 'p', 'iterations', 'other_coeff']
    rows = []

    for metric in required_metrics:
        header.append(f"mean_{metric}")
        header.append(f"best_{metric}")
        header.append(f"std_{metric}")
        header.append(f"sem_{metric}")

    for result in results:
        # curr_row = [result['psx'], result['prior'], result['p'], result['lr'], result['folds'], result['iter']]
        curr_row = [result['psx'], result['lr'], result['folds'], result['p'], result['iter'], result['other_coeff']]
        for metric in required_metrics:
            curr_row.append(float(round(result[f"mean_{metric}"], 3)))
            curr_row.append(float(round(max(result[metric]), 3)))
            curr_row.append(float(round(result[f"std_{metric}"], 4)))
            curr_row.append(float(round(result[f"sem_{metric}"], 4)))
        rows.append(curr_row)

    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)


if __name__ == "__main__":
    input_file = "/Users/asedova/PycharmProjects/01_knodle/data_from_minio/spouse/processed/spouse_ulf_new.json"
    required_metrics = ["f1_avg"]
    output_file = "/Users/asedova/PycharmProjects/01_knodle/acl_results/spouse/ulf_spouse_sgn_new.csv"

    postprocess_json(input_file, required_metrics, output_file)
