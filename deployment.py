import json
import logging
import shutil
from pathlib import Path


def copy_to_production(input_folder, output_folder, filename):
    input_path = Path(input_folder) / Path(filename)
    output_path = Path(output_folder) / Path(filename)
    logging.info("Copying %s to %s", input_path, output_path)
    shutil.copy(input_path, output_path)


def deploy():
    with open('config.json', 'r') as f:
        config = json.load(f)

    deployment_folder = Path(config["prod_deployment_path"])

    model_folder = Path(config['output_model_path'])
    model_name = Path("trainedmodel.pkl")
    copy_to_production(input_folder=model_folder, output_folder=deployment_folder, filename=model_name)

    test_result_filename = Path("latestscore.txt")
    copy_to_production(input_folder=model_folder, output_folder=deployment_folder, filename=test_result_filename)

    data_folder = Path(config['output_folder_path'])
    datset_log_file = Path("ingestedfiles.txt")
    copy_to_production(input_folder=data_folder, output_folder=deployment_folder, filename=datset_log_file)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    deploy()
