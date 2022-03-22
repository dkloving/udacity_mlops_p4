import json
import logging
from pathlib import Path

import pandas as pd
from sklearn import metrics

from dbsetup import ProjectDB


def score_model(test_data, model):
    """Scores the latest model on an input dataframe
    """
    y = test_data["exited"]
    X = test_data[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    y_pred = model.predict(X)
    score = metrics.f1_score(y, y_pred)
    logging.info("Test score: %f", score)
    return score


def score_on_test_file():
    """Scores the latest model on the test dataset in `test_data_path`
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = Path(config['test_data_path']) / Path("testdata.csv")

    logging.info("Reading data from %s", test_data_path)
    test_data = pd.read_csv(test_data_path)

    logging.info("Reading latest model from sqlite")
    db = ProjectDB()
    model_obj = db.get_latest_model()
    model = model_obj['model']

    score = score_model(test_data, model)
    return score


def write_test_score():
    """Scores the latest model on testdata.csv and saves score to a text file in `output_model_path`.
    """
    score = score_on_test_file()

    with open('config.json', 'r') as f:
        config = json.load(f)

    test_result_path = Path(config['output_model_path']) / Path("latestscore.txt")

    logging.info("Writing score to %s", test_result_path)
    with open(test_result_path, 'w') as file:
        file.write(str(score))


if __name__ == "__main__":
    write_test_score()
