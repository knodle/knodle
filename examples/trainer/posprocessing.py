import os
import json
import csv
from typing import List


def postprocess_json(input_file: str, required_metrics: List, output_file: str) -> None:

    with open(input_file, encoding="utf-8") as file:
        results = json.load(file)

    header = ['psx', 'prior', 'p', 'lr', 'folds', 'iterations']
    rows = []

    for metric in required_metrics:
        header.append(f"mean_{metric}")
        header.append(f"best_{metric}")
        header.append(f"std_{metric}")
        header.append(f"sem_{metric}")

    for result in results:
        curr_row = [result['psx'], result['prior'], result['p'], result['lr'], result['folds'], result['iter']]
        for metric in required_metrics:
            curr_row.append(float(round(result[f"mean_{metric}"], 3)))
            curr_row.append(float(round(max(result[metric]), 3)))
            curr_row.append(float(round(result[f"std_{metric}"], 4)))
            curr_row.append(float(round(result[f"sem_{metric}"], 4)))
        rows.append(curr_row)

    os.makedirs(output_file, exist_ok=True)

    with open(os.path.join(output_file, "ulf_sms_logreg_10epochs_refactored.csv"), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)


if __name__ == "__main__":
    input_file = "/Users/asedova/PycharmProjects/01_knodle/data_from_minio/sms/processed/sms_ulf_logreg_10exp_40epochs.json"
    required_metrics = ["f1_avg"]
    output_file = "/Users/asedova/PycharmProjects/01_knodle/data_from_minio/sms/processed/"

    postprocess_json(input_file, required_metrics, output_file)
