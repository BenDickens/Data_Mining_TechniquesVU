import pandas as pd
import csv
import matplotlib.pyplot as plt


def get_weekday(population_csv):
    data = pd.read_csv(population_csv)
    data["time"] = pd.to_datetime(data["time"])

    for index, row in data.iterrows():
        weekday = str(row["time"].weekday())
        data.at[index, 'weekday'] = weekday

    data.to_csv(f"./plots/population+weekend.csv")


def get_average_for_weekday(population_weekend_csv):
    data = pd.read_csv(population_weekend_csv)
    means = list()
    for i in range(7):
        day_data = data[data["weekday"] == i]
        mean = day_data["mood"].mean()
        means.append((i, round(mean, 3)))

    df_mean = pd.DataFrame(means, columns=["day", "mean"])
    translation = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df_mean.day = df_mean.day.map(translation)

    df_mean.to_csv('./plots/average_by_weekday.csv')

def plot_mean_weekday(average_weekday_csv):
    x = []
    y = []

    with open(average_weekday_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        for row in plots:
            x.append(row[1])
            y.append(float(row[2]))

    plt.plot(x, y)
    plt.xlabel('Day of the week')
    plt.ylabel('Average mood score of the population')
    plt.title('Average mood per weekday of the entire population')
    plt.savefig("average_by_weekday.png")

average_weekday_csv = '../plots/average_by_weekday.csv'
plot_mean_weekday(average_weekday_csv)