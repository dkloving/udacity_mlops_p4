import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # check for datasets
    input_filenames = list(Path(input_folder_path).glob("*.csv"))
    logging.info("Found %i datasets", len(input_filenames))

    # compile them together,
    datasets = [pd.read_csv(file) for file in input_filenames]
    combined_dataset = pd.concat(datasets, axis=0)
    logging.info("Rows in total dataset: %i, columns: %i", len(combined_dataset), len(combined_dataset.columns))

    # De-duplication
    combined_dataset = combined_dataset.drop_duplicates()
    logging.info("After removing duplicates: %i", len(combined_dataset))

    # write to an output file
    output_filename = Path(output_folder_path) / Path("finaldata.csv")
    logging.info("Writing dataset to %s", output_filename)
    combined_dataset.to_csv(output_filename, index=False)

    # log input files used
    log_filename = Path(output_folder_path) / Path("ingestedfiles.txt")
    logging.info("Writing input file log to %s", log_filename)
    with open(log_filename, 'w') as lf:
        lf.write(str(datetime.now()))
        lf.write('\n')
        for fn in input_filenames:
            lf.write(str(fn))
            lf.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    merge_multiple_dataframe(input_folder_path, output_folder_path)
