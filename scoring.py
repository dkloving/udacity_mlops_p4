import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn import metrics


def score_model(test_data_folder, trained_model_folder):
    dataset_file_path = Path(test_data_folder) / Path("testdata.csv")
    logging.info("Reading data from %s", dataset_file_path)
    test_data = pd.read_csv(dataset_file_path)

    model_file_path = Path(trained_model_folder) / Path("trainedmodel.pkl")
    logging.info("Reading model from %s", trained_model_folder)
    with open(model_file_path, 'rb') as file:
        clf = pickle.load(file)

    y = test_data["exited"]
    X = test_data[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    y_pred = clf.predict(X)
    score = metrics.f1_score(y, y_pred)
    logging.info("Test score: %f", score)

    output_filename = Path(trained_model_folder) / Path("latestscore.txt")
    logging.info("Writing score to %s", output_filename)
    with open(output_filename, 'w') as file:
        file.write(str(score))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    trained_model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])

    score_model(test_data_path, trained_model_path)
