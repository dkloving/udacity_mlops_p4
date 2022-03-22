import io
import json
import logging
import pickle
import sqlite3
from datetime import datetime

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
            datasetid INT PRIMARY KEY,
            input_filenames TEXT,
            dataset TEXT,
            creation_time TEXT
        );""")

        # create models table
        cls._cursor.execute("""CREATE TABLE IF NOT EXISTS models(
            modelid INT PRIMARY KEY,
            model_pkl BLOB,
            training_dataset INT,
            creation_date TEXT,
            FOREIGN KEY(training_dataset) REFERENCES datasets(datasetid)
        );""")

        # create scores table
        cls._cursor.execute("""CREATE TABLE IF NOT EXISTS scores(
            scoreid INT PRIMARY KEY,
            modelid INT,
            f1 FLOAT,
            creation_date TEXT,
            FOREIGN KEY(modelid) REFERENCES models(modelid)
        );""")

    @classmethod
    def insert_dataset(cls, input_filenames, dataset):
        cls._cursor.execute("SELECT * FROM datasets")
        result = cls._cursor.fetchall()
        next_id = len(result)

        cls._cursor.execute(
            "INSERT INTO datasets VALUES(?, ?, ?, ?);",
            (next_id, input_filenames, dataset, str(datetime.now()))
        )
        cls._connection.commit()

    @ classmethod
    def get_latest_dataset(cls):
        dset = {'id': None, 'file_list': None, 'data': None, 'creation_date': None}
        cls._cursor.execute("SELECT * FROM datasets ORDER BY datasetid DESC;")
        latest_dataset = cls._cursor.fetchone()
        if latest_dataset is not None:
            dset['id'] = latest_dataset[0]
            dset['file_list'] = latest_dataset[1].split(',')
            dset['data'] = pd.read_csv(io.StringIO(latest_dataset[2]), index_col=0)
            dset['creation_date'] = latest_dataset[3]
        return dset

    @classmethod
    def insert_model(cls, model, training_dataset_id):
        cls._cursor.execute("SELECT * FROM models")
        result = cls._cursor.fetchall()
        next_id = len(result)

        model_txt = pickle.dumps(model)

        cls._cursor.execute(
            "INSERT INTO models VALUES(?, ?, ?, ?);",
            (next_id, model_txt, training_dataset_id, str(datetime.now()))
        )
        cls._connection.commit()

    @ classmethod
    def get_latest_model(cls):
        model = {'id': None, 'model_pkl': None, 'training_dataset': None, 'creation_date': None}
        cls._cursor.execute("SELECT * FROM models ORDER BY modelid DESC;")
        latest_dataset = cls._cursor.fetchone()
        if latest_dataset is not None:
            model['id'] = latest_dataset[0]
            model['model'] = pickle.loads(latest_dataset[1])
            model['training_dataset'] = latest_dataset[2]
            model['creation_date'] = latest_dataset[3]
        return model

    @classmethod
    def insert_score(cls, score, model_id):
        cls._cursor.execute("SELECT * FROM scores")
        result = cls._cursor.fetchall()
        next_id = len(result)

        cls._cursor.execute(
            "INSERT INTO scores VALUES(?, ?, ?, ?);",
            (next_id, model_id, score, str(datetime.now()))
        )
        cls._connection.commit()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    db = ProjectDB()
