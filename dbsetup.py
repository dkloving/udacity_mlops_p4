import io
import json
import logging
import pickle
import sqlite3

import pandas as pd


class ProjectDB:

    _instance = None
    _connection = None
    _cursor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectDB, cls).__new__(cls)
            with open('config.json', 'r') as f:
                config = json.load(f)
            logging.info("Using database at %s", config['db_path'])
            cls._connection = sqlite3.connect(config['db_path'])
            cls._cursor = cls._connection.cursor()
            cls.setup()
        return cls._instance

    @classmethod
    def setup(cls):

        # create dataset table
        cls._cursor.execute("""CREATE TABLE IF NOT EXISTS datasets(
            id INT PRIMARY KEY,
            input_filenames TEXT,
            dataset TEXT,
            creation_time TEXT default CURRENT_TIMESTAMP
        );""")

        # create models table
        cls._cursor.execute("""CREATE TABLE IF NOT EXISTS models(
            id INT PRIMARY KEY,
            model_pkl BLOB,
            training_dataset INT,
            creation_date TEXT default CURRENT_TIMESTAMP,
            FOREIGN KEY(training_dataset) REFERENCES datasets(datasetid)
        );""")

        # create diagnostics table
        cls._cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics(
            id INT PRIMARY KEY,
            datasetid INT,
            modelid INT,
            lastmonth_mean FLOAT,
            lastmonth_median FLOAT,
            lastmonth_stddev FLOAT,
            lastmonth_missing FLOAT,
            lastyear_mean FLOAT,
            lastyear_median FLOAT,
            lastyear_stddev FLOAT,
            lastyear_missing FLOAT,
            number_of_employees_mean FLOAT,
            number_of_employees_median FLOAT,
            number_of_employees_stddev FLOAT,
            number_of_employees_missing FLOAT,
            ingestion_time FLOAT,
            training_time FLOAT,
            f1_score FLOAT,
            packages TEXT,
            creation_date TEXT default CURRENT_TIMESTAMP,
            FOREIGN KEY(modelid) REFERENCES models(modelid),
            FOREIGN KEY(datasetid) REFERENCES datasets(datasetid)
        );""")

    @classmethod
    def insert_dataset(cls, input_filenames, dataset):
        logging.info("Writing dataset to db")
        cls._cursor.execute("SELECT * FROM datasets")
        result = cls._cursor.fetchall()
        next_id = len(result)

        cls._cursor.execute(
            "INSERT INTO datasets(id, input_filenames, dataset) VALUES(?, ?, ?);",
            (next_id, input_filenames, dataset)
        )
        cls._connection.commit()

    @ classmethod
    def get_latest_dataset(cls):
        dset = {'id': None, 'file_list': None, 'data': None, 'creation_date': None}
        cls._cursor.execute("SELECT * FROM datasets ORDER BY id DESC;")
        latest_dataset = cls._cursor.fetchone()
        if latest_dataset is not None:
            dset['id'] = latest_dataset[0]
            dset['file_list'] = latest_dataset[1].split(',')
            dset['data'] = pd.read_csv(io.StringIO(latest_dataset[2]), index_col=0)
            dset['creation_date'] = latest_dataset[3]
        return dset

    @classmethod
    def insert_model(cls, model, training_dataset_id):
        logging.info("Writing model to db")
        cls._cursor.execute("SELECT * FROM models")
        result = cls._cursor.fetchall()
        next_id = len(result)

        model_txt = pickle.dumps(model)

        cls._cursor.execute(
            "INSERT INTO models(id, model_pkl, training_dataset) VALUES(?, ?, ?);",
            (next_id, model_txt, training_dataset_id)
        )
        cls._connection.commit()

    @ classmethod
    def get_latest_model(cls):
        model = {'id': None, 'model_pkl': None, 'training_dataset': None, 'creation_date': None}
        cls._cursor.execute("SELECT * FROM models ORDER BY id DESC;")
        latest_dataset = cls._cursor.fetchone()
        if latest_dataset is not None:
            model['id'] = latest_dataset[0]
            model['model'] = pickle.loads(latest_dataset[1])
            model['training_dataset'] = latest_dataset[2]
            model['creation_date'] = latest_dataset[3]
        return model

    @classmethod
    def insert_diagnostics(
            cls,
            dataset_id,
            model_id,
            lastmonth_activity_mean,
            lastmonth_activity_median,
            lastmonth_activity_stddev,
            lastmonth_activity_missing,
            lastyear_activity_mean,
            lastyear_activity_median,
            lastyear_activity_stddev,
            lastyear_activity_missing,
            number_of_employees_mean,
            number_of_employees_median,
            number_of_employees_stddev,
            number_of_employees_missing,
            ingestion_time,
            training_time,
            f1_score,
            packages_csv
    ):
        logging.info("Writing diagnostics to db")
        cls._cursor.execute("SELECT * FROM diagnostics ")
        result = cls._cursor.fetchall()
        next_id = len(result)

        cls._cursor.execute(
            """
                INSERT INTO diagnostics(
                    id,
                    datasetid,
                    modelid,
                    lastmonth_mean,
                    lastmonth_median,
                    lastmonth_stddev,
                    lastmonth_missing,
                    lastyear_mean,
                    lastyear_median,
                    lastyear_stddev,
                    lastyear_missing,
                    number_of_employees_mean,
                    number_of_employees_median,
                    number_of_employees_stddev,
                    number_of_employees_missing,
                    ingestion_time,
                    training_time,
                    f1_score,
                    packages
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                next_id,
                dataset_id,
                model_id,
                lastmonth_activity_mean,
                lastmonth_activity_median,
                lastmonth_activity_stddev,
                lastmonth_activity_missing,
                lastyear_activity_mean,
                lastyear_activity_median,
                lastyear_activity_stddev,
                lastyear_activity_missing,
                number_of_employees_mean,
                number_of_employees_median,
                number_of_employees_stddev,
                number_of_employees_missing,
                ingestion_time,
                training_time,
                f1_score,
                packages_csv
            )
        )
        cls._connection.commit()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    db = ProjectDB()
