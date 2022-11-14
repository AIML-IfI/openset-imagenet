import csv
import pandas as pd

# function for reading OSCR data and manipulating column names


def load_oscr_metrics(csv_path):
    """
    Args:
        csv_path (string, Path): Path to the csv file having CCR and FPR values for plotting OSCR curve
    """
    # read csv file
    data = pd.read_csv(csv_path)
    return data


def store_oscr_metrics(csv_path, fpr, ccr, cover):
    """
    Args:
        csv_path (string, Path): Path to the csv file where metrics are stored
    """
    rows = zip(fpr.tolist(), ccr.tolist(), cover.tolist())

    with open(csv_path, 'w') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(('fpr', 'ccr', 'cover'))
        for row in rows:
            writer.writerow(row)
