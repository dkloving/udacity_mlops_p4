import io
import sqlite3

import pandas as pd


class ProjectDB:

    _instance = None
    _connection = None
    _cursor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectDB, cls).__new__(cls)
            cls._connection = sqlite3.connect("project.db")
            cls._cursor = cls._connection.cursor()
        return cls._instance

    @classmethod
    def migrate(cls):

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
            FOREIGN KEY(modelid) REFERENCES models(modelid)
        );""")

    @classmethod
    def insert_dataset(cls, input_filenames, dataset, creation_date):
        cls._cursor.execute("SELECT * FROM datasets")
        result = cls._cursor.fetchall()
        next_id = len(result)

        cls._cursor.execute(
            "INSERT INTO datasets VALUES(?, ?, ?, ?);",
            (next_id, input_filenames, dataset, creation_date)
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


if __name__ == "__main__":
    db = ProjectDB()
    db.migrate()
