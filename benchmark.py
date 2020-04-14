import os
import pandas as pd


def aggregate_data(folderpath):
    for filename in os.listdir(folderpath)[:1]:
        data = pd.read_csv(folderpath + filename)

        # Create patient number column
        patientno = filename.replace("patientAS14.", "").replace(".csv", "")
        data["patient"] = patientno

        # Reorder the columns so patient is in the front
        patient = data["patient"]
        data.drop(labels=["patient"], axis=1, inplace=True)
        data.insert(0, "patient", patient)

        # Split time to get month and day
        for index, row in data.iterrows():
            year, month, day = row["time"].split("-")
            data.at[index, "month"] = month
            data.at[index, "day"] = day

        # Remove the time column and move month and day to the front
        month = data["month"]
        day = data["day"]
        data.drop(labels=["month", "day", "time"], axis=1, inplace=True)
        data.insert(1, "month", month)
        data.insert(2, "day", day)

        # Move mood to last column
        mood = data["mood"]
        data.drop(labels=["mood"], axis=1, inplace=True)
        data.insert(21, "mood", mood)

        # Fill NaN for all columns apart from mood
        columns = list(data.columns[3:21])
        data[columns] = data[columns].fillna(value=0)

        # TODO: Make sure the benchmark takes the day before and not any other day.
        # Create column with benchmark prediction
        data["benchmark"] = data.mood.shift()

        # Drop empty mood and benchmark columns
        data = data[data["mood"].notna()]
        data = data[data["benchmark"].notna()]

        print(data)


# def create_benchmark():

# def evaluate_benchmark():

folderpath = "./patientData/"
aggregate_data(folderpath)
