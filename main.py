import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np


def main():
    dataList = csvToDict("mini-AAPL.csv")

    date = []
    openPrice = []
    for row in dataList:
        openPrice.append(row["Open"])
        date.append(row["Date"])

    X = date
    y = np.array(openPrice, dtype=np.float64).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.20)
    # slrModel = sklearn.LinearRegression() #creates an empty model
    # slrModel.fit(X_train, y_train) #create the model

    
    # geeksforgeeks.org/machine-learning/comprehensive-guide-toclassification-models-in-scikit-learn/
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))
    print(metrics.classification_report(y_test, y_pred_knn))

    
    

def csvToDict(filename):
    dataList = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataList.append(row)
    return dataList




main()

