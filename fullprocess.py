import json
import logging
import logging.config
import pickle
from pathlib import Path

import deployment
import diagnostics
import ingestion
import reporting
import scoring
import training
from dbsetup import ProjectDB


def check_model_exists(db: ProjectDB):
    model_obj = db.get_latest_model()
    exists = model_obj['model'] is not None
    return exists


def check_new_data(config, db: ProjectDB):
    ingested_files = db.get_latest_dataset()['file_list']
    if ingested_files is None:
        return True

    else:
        logging.info("Found %i files were previously ingested.", len(ingested_files))

        source_files = list(Path(config["input_folder_path"]).glob("*.csv"))
        logging.info("Found %s source files.", len(source_files))

        new_files_present = any([file not in ingested_files for file in source_files])
        logging.info("New files present: %s", 'TRUE' if new_files_present else 'FALSE')

        return new_files_present


def check_model_drift(config, db: ProjectDB):
    model_score_path = Path(config["prod_deployment_path"]) / Path("latestscore.txt")
    if not model_score_path.exists():
        return True

    else:
        with open(model_score_path) as f:
            recent_score = json.load(f)['f1_score']

        production_model_path = Path(config['prod_deployment_path']) / Path("trainedmodel.pkl")
        with open(production_model_path, 'rb') as file:
            model = pickle.load(file)

        new_data = db.get_latest_dataset()['data']

        new_score = scoring.score_model(new_data, model)

        drift = new_score < recent_score
        logging.info(
            "New model score (%f) vs previous (%f), drift detected: %s",
            new_score,
            recent_score,
            "TRUE" if drift else "FALSE"
        )

        return drift


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')

    # initialize database
    db = ProjectDB()

    with open('config.json', 'r') as f:
        config = json.load(f)

    model_exists = check_model_exists(db)
    if model_exists:

        new_data = check_new_data(config, db)
        if not new_data:
            # quit()
            pass

        logging.info("Ingesting new data...")
        ingestion.merge_multiple_dataframe()

        model_drift = check_model_drift(config, db)
        if not model_drift:
            # quit()
            pass

    logging.info("Training and Scoring new model...")
    training.train_model()

    logging.info("Diagnostics and Reporting...")
    diagnostics.run()
    reporting.run()

    logging.info("Deploying...")
    deployment.deploy_latest()
