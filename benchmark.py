import os
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import math
import datetime


def create_benchmark(folderpath):
    """
    Loops over the data
    :param folderpath:
    :return: data
    """
    complete_data = pd.DataFrame()

    for filename in os.listdir(folderpath):
        data = pd.read_csv(folderpath + filename)

        # Create patient number column
        patientno = filename.replace("patientAS14.", "").replace(".csv", "")
        data["patient"] = patientno

        # Create column with benchmark prediction
        # Convert time to datetime
        data["time"] = pd.to_datetime(data["time"])
        data["day_dif"] = data.time - data.time.shift()

        data.drop(data[data["day_dif"] != "1 days"].index, inplace=True)
        data["benchmark"] = data.mood.shift()

        # Drop empty mood and benchmark columns
        data = data[data["mood"].notna()]
        data = data[data["benchmark"].notna()]

        # Drop all unnecessary columns
        cols = ["patient", "time", "mood", "benchmark"]
        data = data[cols]

        complete_data = complete_data.append(data)
    complete_data.to_csv('./benchmark_data/benchmark.csv')
    return complete_data

def evaluate_benchmark(folderpath):
    data = create_benchmark(folderpath)
    y_true = data["mood"].to_numpy()
    y_pred = data["benchmark"].to_numpy()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    y_true = data["mood"].astype(int).to_numpy()
    y_pred = data["benchmark"].astype(int).to_numpy()
    accuracy = accuracy_score(y_true, y_pred)

    print(accuracy, rmse)
    return accuracy, rmse


folderpath = "./patientData/"
evaluate_benchmark(folderpath)
