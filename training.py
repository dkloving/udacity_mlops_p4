import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(input_dataset_path, output_model_path):
    dataset_file_path = Path(input_dataset_path) / Path("finaldata.csv")
    logging.info("Reading data from %s", dataset_file_path)
    dataset = pd.read_csv(dataset_file_path)
    y = dataset["exited"]
    X = dataset[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    # use this logistic regression for training
    clf = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=1000, n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )

    # fit the logistic regression to your data
    logging.info("Fitting model.")
    clf.fit(X, y)
    logging.info("Score on training data: %f", clf.score(X, y))

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_file_path = Path(output_model_path) / Path("trainedmodel.pkl")
    logging.info("Saving model to %s", model_file_path)
    with open(model_file_path, 'wb') as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    train_model(dataset_csv_path, model_path)
