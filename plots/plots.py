import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import datetime


def plot_population(population_csv):
    x = []
    y = []

    with open(population_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        for row in plots:
            x.append(row[0])
            y.append(round(float(row[1]), 1))

    x = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in x]
    ax = plt.gca()

    formatter = mdates.DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(formatter)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    plt.plot(x, y)
    plt.xlabel('Date')
    plt.ylabel('Average mood score of the population')
    plt.title('Average mood per day of the entire population')
    plt.savefig("average_mood_population.png")


def run():
    population_csv = "./plots/population.csv"
    plot_population(population_csv)


if __name__ == "__main__":
    run()
