import logging
import os
import subprocess
import timeit

import pandas as pd

from dbsetup import ProjectDB
from scoring import score_model


def model_predictions(dataset, model):
    """Gets model predictions on a dataset.
    NOTE: it really shouldn't be here, but the rubric requires it to be here.
    """
    X = dataset[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    y_pred = model.predict(X)

    return y_pred


def dataframe_summary(dataset) -> list:
    """Column-wise summary statistics
    """
    logging.info("Calculating summary statistics")
    summary = []
    numerical_cols = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]

    for col in numerical_cols:
        summary.append(dataset[col].mean())
        summary.append(dataset[col].median())
        summary.append(dataset[col].std())
    return summary


def check_missing_data(dataset) -> list:
    """Column-wise percent of values missing
    """
    numerical_cols = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]

    na_pcts = []
    for col in numerical_cols:
        pct_na = dataset[col].isna().sum() / len(dataset)
        na_pcts.append(pct_na)
    return na_pcts


def execution_time():
    """Times data ingestion and model training processes
    """
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
    """Creates a dataframe with all pip packages installed, including current and latest-available versions
    """
    logging.info("Checking packages with pip.")
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


def run():
    db = ProjectDB()

    dataset_obj = db.get_latest_dataset()
    dataset = dataset_obj['data']

    model_obj = db.get_latest_model()
    model = model_obj['model']

    score = score_model(dataset, model)
    lm_mean, lm_median, lm_std, ly_mean, ly_median, ly_std, ne_mean, ne_median, ne_std = dataframe_summary(dataset)
    lm_missing, ly_missing, ne_missing = check_missing_data(dataset)
    ingestion_timing, training_timing = execution_time()
    packages_df = outdated_packages_list()

    db.insert_diagnostics(
        dataset_id=dataset_obj['id'],
        model_id=model_obj['id'],
        lastmonth_activity_mean=lm_mean,
        lastmonth_activity_median=lm_median,
        lastmonth_activity_stddev=lm_std,
        lastmonth_activity_missing=lm_missing,
        lastyear_activity_mean=ly_mean,
        lastyear_activity_median=ly_median,
        lastyear_activity_stddev=ly_std,
        lastyear_activity_missing=ly_missing,
        number_of_employees_mean=ne_mean,
        number_of_employees_median=ne_median,
        number_of_employees_stddev=ne_std,
        number_of_employees_missing=ne_missing,
        ingestion_time=ingestion_timing,
        training_time=training_timing,
        f1_score=score,
        packages_csv=packages_df.to_csv()
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
