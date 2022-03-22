import json
import logging
from pathlib import Path

import pandas as pd
from sklearn import metrics

from dbsetup import ProjectDB


def score_model(test_data_path=None):
    with open('config.json', 'r') as f:
        config = json.load(f)

    if not test_data_path:
        test_data_path = Path(config['test_data_path']) / Path("testdata.csv")

    logging.info("Reading data from %s", test_data_path)
    test_data = pd.read_csv(test_data_path)

    logging.info("Reading latest model from sqlite")
    db = ProjectDB()
    model_obj = db.get_latest_model()
    clf = model_obj['model']

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

    logging.info("Writing score to sqlite")
    db = ProjectDB()
    model_obj = db.get_latest_model()
    db.insert_score(score=score, model_id=model_obj['id'])


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    write_score()
