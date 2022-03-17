import json
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, make_response

from diagnostics import model_predictions, dataframe_summary, execution_time, check_missing_data, outdated_packages_list
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

prediction_model_path = Path(config['output_model_path']) / Path("trainedmodel.pkl")


def read_pandas(filename):
    df = pd.read_csv(filename)
    return df


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    filename = request.args.get("filename")
    df = read_pandas(filename)
    preds = model_predictions(df, prediction_model_path)
    preds = [str(i) for i in preds]
    response = make_response(jsonify({"predictions": preds}))
    return response


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    score = score_model()
    response = make_response(jsonify({"score": score}))
    return response


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    summary = dataframe_summary()
    response = make_response(jsonify({"summary": summary}))
    return response


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def package_versions():
    timing = execution_time()
    missing_data = check_missing_data()
    packages = outdated_packages_list()
    packages = packages.to_json(index=False, orient='split')
    response = make_response(jsonify({
        "timing": timing,
        "missing_data": missing_data,
        "packages": packages
    }))
    return response


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
