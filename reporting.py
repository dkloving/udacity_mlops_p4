"""
Functionality to plot and save a confusion matrix of the latest model on test data
"""

import json
import logging
import logging.config
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dbsetup import ProjectDB
from diagnostics import model_predictions


def make_confusion_matrix(test_data, model, save_location=None):
    """Plots a Confusion Matrix and optionally saves it to disk
    """
    test_predictions = model_predictions(test_data, model)
    test_true = test_data["exited"]

    cm = confusion_matrix(test_true, test_predictions)
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()

    if save_location:
        figure_path = save_location / Path("confusionmatrix.png")
        logging.info("Writing confusion matrix to %s", figure_path)
        plt.savefig(figure_path)


def run():
    with open('config.json', 'r') as f:
        config = json.load(f)
    output_folder = Path(config["output_model_path"])

    db = ProjectDB()
    model_obj = db.get_latest_model()
    model = model_obj['model']

    test_data_path = Path(config['test_data_path']) / Path("testdata.csv")
    logging.info("Reading data from %s", test_data_path)
    test_data = pd.read_csv(test_data_path)

    make_confusion_matrix(test_data, model, output_folder)


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')
    run()
