"""
Provides functionality for 'deploying' an ML model.
"""

import json
import logging
import logging.config
import pickle
from pathlib import Path

from dbsetup import ProjectDB


def deploy_latest():
    """Writes latest model and associated reporting objects to a deployment folder
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    deployment_folder = Path(config["prod_deployment_path"])

    logging.info("Logging production artifacts to %s", deployment_folder)

    db = ProjectDB()

    model_obj = db.get_latest_model()
    dataset_obj = db.get_latest_dataset()
    diagnostics = db.get_diagnostics(dataset_obj['id'])

    model_output_name = deployment_folder / Path("trainedmodel.pkl")
    with open(model_output_name, 'wb') as file:
        pickle.dump(model_obj['model'], file)

    test_result_name = deployment_folder / Path("latestscore.txt")
    with open(test_result_name, 'w') as file:
        file.write(str(diagnostics['f1_score']))

    dataset_log_name = deployment_folder / Path("ingestedfiles.txt")
    with open(dataset_log_name, 'w') as file:
        file.write(','.join(dataset_obj['file_list']))


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    deploy_latest()
