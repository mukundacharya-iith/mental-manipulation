import os
import csv
import numpy as np
from datetime import datetime


def init_results_dir():
    os.makedirs("results", exist_ok=True)


def log_to_file(message, filename):
    with open(filename, "a") as f:
        f.write(message + "\n")


def save_metrics(result, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "dataset",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro"
            ])

        writer.writerow([
            datetime.now(),
            result["name"],
            result["accuracy"],
            result["f1_macro"],
            result["f1_weighted"],
            result["precision_macro"],
            result["recall_macro"]
        ])


def save_confusion_matrix(cm, filename):
    np.savetxt(filename, cm, delimiter=",", fmt="%d")


def save_report(report, filename):
    with open(filename, "w") as f:
        f.write(report)