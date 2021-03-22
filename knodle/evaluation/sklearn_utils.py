from typing import Dict


def sklearn_report_to_knodle_report(sklearn_report: Dict, prefix: str = None):
    if prefix is None:
        report = {
            f"accuracy": sklearn_report["accuracy"],
            f"macro_f1": sklearn_report["macro avg"]["f1-score"],
            f"weighted_f1": sklearn_report["weighted avg"]["f1-score"],
        }
    else:
        report = {
            f"{prefix}accuracy": sklearn_report["accuracy"],
            f"{prefix}macro_f1": sklearn_report["macro avg"]["f1-score"],
            f"{prefix}weighted_f1": sklearn_report["weighted avg"]["f1-score"],
        }
    return report
