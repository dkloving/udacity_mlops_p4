import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from diagnostics import model_predictions


def score_model(test_file_path, model_file_path, output_path):
    test_df = pd.read_csv(test_file_path)
    test_predictions = model_predictions(test_df, model_file_path)
    test_true = test_df["exited"]

    cm = confusion_matrix(test_true, test_predictions)
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    figure_path = output_path / Path("confusionmatrix.png")
    logging.info("Writing confusion matrix to %s", figure_path)
    plt.savefig(figure_path)


def run():
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_csv_path = Path(config['test_data_path']) / Path("testdata.csv")
    saved_model_path = Path(config['prod_deployment_path']) / Path("trainedmodel.pkl")
    output_folder = Path(config["output_model_path"])

    score_model(test_csv_path, saved_model_path, output_folder)


if __name__ == '__main__':
    run()
