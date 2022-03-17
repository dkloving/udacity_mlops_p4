import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn import metrics


def score_model():
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = Path(config['test_data_path']) / Path("testdata.csv")
    trained_model_path = Path(config['output_model_path']) / Path("trainedmodel.pkl")

    logging.info("Reading data from %s", test_data_path)
    test_data = pd.read_csv(test_data_path)

    logging.info("Reading model from %s", trained_model_path)
    with open(trained_model_path, 'rb') as file:
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
    return score


def write_score():
    score = score_model()

    with open('config.json', 'r') as f:
        config = json.load(f)

    test_result_path = Path(config['output_model_path']) / Path("latestscore.txt")

    logging.info("Writing score to %s", test_result_path)
    with open(test_result_path, 'w') as file:
        file.write(str(score))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    write_score()
