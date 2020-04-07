import pandas as pd
import numpy as np  



def read():
    data = pd.read_csv(open('./dataset_mood_smartphone.csv'))
    patients = np.unique(list(data.id))
    counter = 0
    for i in patients:
        patDat = data.loc[data.id==i]
        del patDat['id']
        del patDat['Unnamed: 0']
        patDat['time'] = pd.to_datetime(patDat['time']).dt.date
        #patDat.index = pd.to_datetime(patDat['time'])
        #del patDat['time']
        p = patDat.variable
        meanVals = ['mood','activity','circumplex.arousal','circumplex.valence']
        meanTable = patDat.loc[p.isin(meanVals)]
        sumTable = patDat.loc[~p.isin(meanVals)]
        cleanMean = meanTable.pivot_table(index="time", columns="variable", values="value", aggfunc=np.mean)
        cleanSum = sumTable.pivot_table(index="time", columns="variable", values="value", aggfunc=np.sum)
        cleanPat = cleanMean.join(cleanSum)
        cleanPat.to_csv("./patientData/patient" + i + ".csv")



def pivotFunc(x):
    return x[0]
        

if __name__ == '__main__':
    read()
