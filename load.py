import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def read():
    data = pd.read_csv(open("./dataset_mood_smartphone.csv"))
    patients = np.unique(list(data.id))
    for i in patients:
        patDat = data.loc[data.id == i]
        del patDat['id']
        del patDat['Unnamed: 0']
        patDat['time'] = pd.to_datetime(patDat['time']).dt.date
        #patDat.index = pd.to_datetime(patDat['time']).dt.date
        #del patDat['time']
        p = patDat.variable
        meanVals = ['mood', 'activity', 'circumplex.arousal', 'circumplex.valence']
        meanTable = patDat.loc[p.isin(meanVals)]
        sumTable = patDat.loc[~p.isin(meanVals)]
        cleanMean = meanTable.pivot_table(index="time", columns="variable", values="value", aggfunc=np.mean)
        cleanSum = sumTable.pivot_table(index="time", columns="variable", values="value", aggfunc=np.sum)
        cleanPat = cleanMean.join(cleanSum)
        cleanPat.to_csv("./patientData/patient" + i + ".csv")

def clean(data):
    data2 = data[(data['time'] > '2014-03-01') & (data['time'] < '2014-05-01') ]
    cleaner = pd.DataFrame(data2['mood'],columns=['mood'])
    cleaner.index = pd.to_datetime(data2['time'])
    pls = cleaner.index.to_series().diff()
    #Raises exception if timeseries has a gap larger than 1 day 
    if pls.max() > pd.Timedelta(1,'D'):
        print("ERROR: Time Series has gap of more than one day")
        raise Exception 
    #A check on the number of mood rows over 2 with NaN seems appropriate but none of the passing datasets have this    
    cleaner['mood'] = cleaner['mood'].interpolate()
    #It is only possible the first or last columns do not have a value after interpolation
    cleaner.dropna(inplace=True)
    #print("mood values null: ",cleaner.isnull().sum())

    return cleaner
    #print(cleaner)
    #for d in cleaner.itertuples():
     #   previous = d.index



    return data

def order_AR(data): 
    ar_data = data
    #https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    result = adfuller(ar_data['mood'].dropna())
    
    print("ADF: ",result[0])
    print("pval: ",result[1])
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(ar_data['mood']); axes[0, 0].set_title('Original Series')
    plot_acf(ar_data['mood'], ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(ar_data['mood'].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(ar_data['mood'].diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(ar_data['mood'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(ar_data['mood'].diff().diff().dropna(), ax=axes[2, 1])

    plt.show()


def ARIMA(data):
    #ass = data['mood'].copy().shift(-1)
    #ar_data = pd.DataFrame(data['mood'],columns=['mood'])
    #ar_data.index = data['time']
    ar_data = data
    result_fuller = adfuller(ar_data['mood'].dropna())
    if result_fuller[1] > 0.05: 
        print("P-Value suggests time series is not stationary")
        raise Exception
    return ar_data
    #print(ar_data)
    #order_AR(ar_data)
    #print(ar_data)
    #plt.plot(ar_data['mood'])
    #plt.show()


if __name__ == '__main__':
    patientNo = "AS14.31"
    data = pd.read_csv(open("./patientData/patient" + patientNo + ".csv"))
    data = clean(data)
    ARIMA(data)


    for i in range(1,34):
        patientNo = str(i)
        patient = "AS14." + patientNo.zfill(2)
        try: 
            print("patient: ", patientNo)
            data = pd.read_csv(open("./patientData/patient" + patient + ".csv"))
            try: data_cleaned = clean(data)
            except: continue
            try: arima = ARIMA(data_cleaned)
            except: continue

            arima.to_csv("./patientDataARCleaned/patient" + patientNo + ".csv")
            #ARIMA(data)
        except: print("ERROR: Patient " + str(i) + " does not exist")


