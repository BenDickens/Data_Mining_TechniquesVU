import os
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import math


def create_benchmark(folderpath, benchmark_filepath):
    """
    Loops over the data in the folder path. Adds a column with patient number, based on the filename.
        Checks whether the dates are consecutive, and only adds the row to the file if it is.
        Creates benchmark from the previous day. Removes all unnecessary columns before writing
        to a new file in benchmark_filepath.

    :param folderpath: path to the folder that contains all patient data.
    :param benchmark_filepath: path to the file to which the output should be written.
    """
    complete_data = pd.DataFrame()

    for filename in os.listdir(folderpath):
        data = pd.read_csv(folderpath + filename)

        # Create patient number column
        patientno = filename.replace("patientAS14.", "").replace(".csv", "")
        data["patient"] = patientno

        # Create column with benchmark prediction
        data["time"] = pd.to_datetime(data["time"])  # Convert time to datetime
        data["day_dif"] = data.time - data.time.shift()  # New column with the number of days in between.

        # Drop all rows with more than one day difference
        data.drop(data[data["day_dif"] != "1 days"].index, inplace=True)

        data["benchmark"] = data.mood.shift()

        # Drop empty mood and benchmark columns
        data = data[data["mood"].notna()]
        data = data[data["benchmark"].notna()]

        # Drop all unnecessary columns
        cols = ["patient", "time", "mood", "benchmark"]
        data = data[cols]

        complete_data = complete_data.append(data)
    complete_data.to_csv(benchmark_filepath)


def evaluate_benchmark(benchmark_filepath):
    """
    Evaluates the benchmark. Get accuracy and root mean square error (RSME) scores.
    :param benchmark_filepath: path to the benchmark.
    :return: accuracy, rmse
    """
    data = pd.read_csv(benchmark_filepath)

    y_true = data["mood"].astype(int).to_numpy()
    y_pred = data["benchmark"].astype(int).to_numpy()
    accuracy = accuracy_score(y_true, y_pred)

    y_true = data["mood"].to_numpy()
    y_pred = data["benchmark"].to_numpy()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Accuracy score benchmark: {accuracy}. \nRMSE score benchmark: {rmse}.")
    return accuracy, rmse


def run():
    folderpath = "./patientData/"
    benchmark_filepath = "./benchmarkData/benchmark.csv"
    create_benchmark(folderpath, benchmark_filepath)
    evaluate_benchmark(benchmark_filepath)


if __name__ == "__main__":
    run()
