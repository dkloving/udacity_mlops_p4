import json
import logging
from pathlib import Path

import deployment
import diagnostics
import ingestion
import reporting
import scoring
import training


def check_model_exists(config):
    trained_model_path = Path(config['output_model_path']) / Path("trainedmodel.pkl")
    return trained_model_path.exists()


def check_new_data(config):
    ingested_files_path = Path(config["output_folder_path"]) / Path("ingestedfiles.txt")
    if not ingested_files_path.exists():
        return True

    else:
        with open(ingested_files_path) as f:
            ingested_files = f.readlines()
        logging.info("Found %i files were previously ingested.", len(ingested_files) - 1)

        source_files = list(Path(config["input_folder_path"]).glob("*.csv"))
        logging.info("Found %s source files.", len(source_files))

        new_files_present = any([file not in ingested_files for file in source_files])
        logging.info("New files present: %s", 'TRUE' if new_files_present else 'FALSE')
        return new_files_present


def check_model_drift(config):
    model_score_path = Path(config["prod_deployment_path"]) / Path("latestscore.txt")
    if not model_score_path.exists():
        return True

    else:
        with open(model_score_path) as f:
            recent_score = float(f.read())

        new_data_path = Path(config['output_folder_path']) / Path("finaldata.csv")
        new_score = scoring.score_model(test_data_path=new_data_path)
        drift = new_score < recent_score
        logging.info(
            "New model score (%f) vs previous (%f), drift detected: %s",
            new_score,
            recent_score,
            "TRUE" if drift else "FALSE"
        )

        return drift


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s | %(module)s:%(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    with open('config.json', 'r') as f:
        config = json.load(f)

    model_exists = check_model_exists(config)
    if model_exists:

        new_data = check_new_data(config)
        if not new_data:
            quit()

        logging.info("Ingesting new data...")
        ingestion.merge_multiple_dataframe()

        model_drift = check_model_drift(config)
        if not model_drift:
            quit()

    logging.info("Training and Scoring new model...")
    training.train_model()
    scoring.write_score()

    logging.info("Deploying...")
    deployment.deploy()

    logging.info("Diagnostics and Reporting...")
    diagnostics.run()
    reporting.run()
