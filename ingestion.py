import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from dbsetup import ProjectDB


def merge_multiple_dataframe(write_db=True, write_file=False):
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_filenames = list(Path(config['input_folder_path']).glob("*.csv"))
    output_filename = Path(config['output_folder_path']) / Path("finaldata.csv")
    log_filename = Path(config['output_folder_path']) / Path("ingestedfiles.txt")

    # check for datasets
    logging.info("Found %i datasets", len(input_filenames))

    # compile them together,
    datasets = [pd.read_csv(file) for file in input_filenames]
    combined_dataset = pd.concat(datasets, axis=0)
    logging.info("Rows in total dataset: %i, columns: %i", len(combined_dataset), len(combined_dataset.columns))

    # De-duplication
    combined_dataset = combined_dataset.drop_duplicates()
    logging.info("After removing duplicates: %i", len(combined_dataset))

    # write to an output file
    logging.info("Writing dataset to %s", output_filename)
    combined_dataset.to_csv(output_filename, index=False)

    if write_file:
        # log input files used
        logging.info("Writing input file log to %s", log_filename)
        with open(log_filename, 'w') as lf:
            lf.write(str(datetime.now()))
            lf.write('\n')
            for fn in input_filenames:
                lf.write(str(fn))
                lf.write('\n')

    if write_db:
        # write to db
        db = ProjectDB()
        input_filenames = ','.join(map(str, input_filenames))
        dataset = combined_dataset.to_csv()
        db.insert_dataset(
            input_filenames=input_filenames,
            dataset=dataset
        )


if __name__ == '__main__':
    merge_multiple_dataframe(write_db=False)
