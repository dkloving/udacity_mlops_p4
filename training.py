"""
Provides functionality for training an ML model.
To meet rubric requirements, this file can be ran independently to pickle and
write the model to disk as `trainedmodel.pkl`.
"""

import json
import logging
import logging.config
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from dbsetup import ProjectDB


def train_model(write_db=True, write_file=False):
    """Train a logistic regression classifier on the latest dataset and save it.

    :param write_db: write model to database
    :param write_file: write model to file
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    logging.info("Reading latest data from sqlite")
    db = ProjectDB()
    dataset_obj = db.get_latest_dataset()
    dataset = dataset_obj['data']

    y = dataset["exited"]
    X = dataset[[
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]]

    clf = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100, n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )

    logging.info("Fitting model.")
    clf.fit(X, y)
    logging.info("Score on training data: %f", clf.score(X, y))

    if write_file:
        model_path = Path(config['output_model_path']) / Path("trainedmodel.pkl")
        logging.info("Saving model to %s", model_path)
        with open(model_path, 'wb') as file:
            pickle.dump(clf, file)

    if write_db:
        db.insert_model(model=clf, training_dataset_id=dataset_obj['id'])


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    train_model(write_db=False, write_file=True)
