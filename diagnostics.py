import json
import logging
import os
import pickle
import subprocess
import timeit
from pathlib import Path

import pandas as pd


def model_predictions(dataset, model_path):
    logging.info("Reading model from %s and getting predictions", model_path)
    with open(model_path, 'rb') as file:
        clf = pickle.load(file)

    X = dataset[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    y_pred = clf.predict(X)

    return y_pred


def dataframe_summary() -> list:
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_path = Path(config["output_folder_path"]) / Path("finaldata.csv")

    logging.info("Calculating summary statistics for %s", dataset_path)
    dataset = pd.read_csv(dataset_path)
    summary = []
    numerica_cols = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]

    for col in numerica_cols:
        summary.append(dataset[col].mean())
        summary.append(dataset[col].median())
        summary.append(dataset[col].std())
    return summary


def check_missing_data() -> list:
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_path = Path(config["output_folder_path"]) / Path("finaldata.csv")

    logging.info("Checking for missing data in %s", dataset_path)
    dataset = pd.read_csv(dataset_path)
    na_pcts = []
    for col in dataset.columns:
        pct_na = dataset[col].isna().sum() / len(dataset)
        na_pcts.append(pct_na)
    return na_pcts


def execution_time():
    logging.info("Timing data ingestion.")
    start_time = timeit.default_timer()
    os.system("python ingestion.py")
    ingestion_timing = timeit.default_timer() - start_time

    logging.info("Timing model training.")
    start_time = timeit.default_timer()
    os.system("python training.py")
    training_timing = timeit.default_timer() - start_time

    return ingestion_timing, training_timing


def outdated_packages_list():
    current = subprocess.run(['pip', 'freeze'], check=True, stdout=subprocess.PIPE, text=True).stdout
    df = pd.DataFrame(columns=["Package", "Installed", "Latest"])
    for c in current.splitlines():
        package = c.split('==')[0]
        installed = c.split('==')[1]

        pip_result = str(subprocess.run(['pip', 'install', f"{package}==nothing"], capture_output=True, text=True))
        versions_start = pip_result.find("(from versions: ") + 16  # +16 to exclude the "(from versions: " text
        versions_end = pip_result.find(')')
        versions = pip_result[versions_start:versions_end]
        versions = versions.replace(',', '').split(' ')
        latest_version = versions[-1].strip()

        df.loc[len(df)] = (package, installed, latest_version)
    return df


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    saved_model_path = Path(config['prod_deployment_path']) / Path("trainedmodel.pkl")
    dataset_path = Path(config["output_folder_path"]) / Path("finaldata.csv")
    dataset = pd.read_csv(dataset_path)

    model_predictions(dataset, saved_model_path)
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()
