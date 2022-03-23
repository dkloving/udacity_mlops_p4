"""
Provides a web-based endpoint for serving the model
"""

import json
import logging.config
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, make_response

from diagnostics import model_predictions

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'  # secrets in version control are bad. Who put this here?

with open('config.json', 'r') as f:
    config = json.load(f)

production_path = Path(config['prod_deployment_path'])


def read_pandas(filename):
    df = pd.read_csv(filename)
    return df


def get_model():
    model_path = production_path / "trainedmodel.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def get_diagnostics_data():
    diagnostics_path = production_path / Path("latestscore.txt")
    with open(diagnostics_path) as file:
        diagnostics_data = json.load(file)
    return diagnostics_data


def get_summary_data():
    summary_path = production_path / Path("dataset_summary.csv")
    summary_df = pd.read_csv(summary_path, index_col=0)
    return summary_df


def get_package_data():
    package_path = production_path / Path("packages.csv")
    package_df = pd.read_csv(package_path, index_col=0)
    return package_df


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """Takes a filename as input and returns model predictions on that file
    """
    filename = request.args.get("filename")
    df = read_pandas(filename)
    model = get_model()
    preds = model_predictions(df, model)
    preds = [str(i) for i in preds]
    response = make_response(jsonify({"predictions": preds}))
    return response


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """Returns production model score on test set that was generated at deployment time
    """
    score = get_diagnostics_data()['f1_score']
    response = make_response(jsonify({"score": score}))
    return response


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """Returns summary statistics about the dataset used to train the current model
    """
    summary = get_summary_data().to_json(index=False, orient='split')
    response = make_response(jsonify({"summary": summary}))
    return response


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def package_versions():
    """Returns diagnostic data generated during training and deployment of the current model.
    """
    diagnostics = get_diagnostics_data()
    timing = {k: diagnostics[k] for k in diagnostics if k != "f1_score"}

    summary_df = get_summary_data()
    missing_data = summary_df.loc["missing"]
    missing_data = missing_data.to_json(index=False, orient='split')

    packages = get_package_data()
    packages = packages.to_json(index=False, orient='split')

    response = make_response(jsonify({
        "timing": timing,
        "missing_data": missing_data,
        "packages": packages
    }))
    return response


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
