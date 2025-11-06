import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import csv
import numpy


def main():
    dataList = csvToDict("mini-AAPL.csv")

    date = []
    opend = []
    for row in dataList:
        opend.append(row["Open"])
        date.append(row["Date"])

    X = [opend]
    y = date

    X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y, random_state=42, test_size = 0.20)
    # slrModel = sklearn.LinearRegression() #creates an empty model
    # slrModel.fit(X_train, y_train) #create the model

    model = sklearn.LinearRegression()
    model.fit(X_train, y_train)

    

def csvToDict(filename):
    dataList = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataList.append(row)
    return dataList




main()

