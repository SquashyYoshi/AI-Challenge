import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import csv
import numpy as np


def main():
    dataList = csvToDict("AAPL.csv")

    dates = []
    openPrice = []
    for row in dataList:
        openPrice.append(float(row["Open"]))
        dates.append(row["Date"])
    dateNums = dateToNum(dates)
    X = np.array(dateToNum(dates)).reshape(-1, 1) #sklearn expects a 2d array, we need to convert to 2d
    y = openPrice

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.20)
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train) #trains the model
    y_pred = knn.predict(X_test)
    print("mean error: "+ float(metrics.mean_absolute_percentage_error(y_test, y_pred)*100) +"%")
    print("R2:"+ metrics.r2_score(y_test, y_pred))

#makes each str in dates be represented as number of days after startDate
def dateToNum(dates):
    #30.41 is average number of days in a month
    dateNums = [] #0 represents first date in dataset, all other numbers represent amount of days after the start day
    dateNums.append(0)
    startNum = int(round((getMonth(dates[0]) * 30.41) + getDay(dates[0]) + (getYear(dates[0]) * 365)))
    
    for date in dates:
        if date == dates[0]:
            continue
        thisDateNum = int(round((getMonth(date) * 30.41) + getDay(date) + (getYear(date) * 365))) - startNum
        if thisDateNum < 0:
            print(f"****ERROR: Calculated negative date number {thisDateNum}, ignoring data point**** ")
        else:
            dateNums.append(thisDateNum)
            #print(f"Added {thisDateNum} to dateNums")
    return dateNums
        
def getMonth(date):
    #print(f"Month num {int(date[3:5])}")
    return int(date[3:5])

def getDay(date):
    #print(f"Day num {int(date[0:2])}")
    return int(date[0:2])

def getYear(date):
    #print(f"Year num {int(date[6:])}")
    return int(date[6:])  

def csvToDict(filename):
    dataList = []
    with open(filename, 'r') as file:
        print("Opened CSV file")
        reader = csv.DictReader(file)
        for row in reader:
            dataList.append(row)
    print("Closed CSV file")
    return dataList




main()

