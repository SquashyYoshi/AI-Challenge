from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
import csv
import sys
import numpy as np

FILENAME = "main.py"
DATASET_PATH = "stock_market_data/forbes2000/csv/"
def main():
    result = parseArgs()
    if result[0] == "search":
        #TODO search function
        pass
    elif result[0] == "error":
        return
    else:
        result = result[0]
    print("Preparing data...")
    dataList = csvToDict(result)
    dates = []
    openPrice = []
    yesterdayPrices = []
    lastOpen = float(dataList[0]["Open"])
    for row in dataList:
        openPrice.append(float(row["Open"]))
        dates.append(row["Date"])
        yesterdayPrices.append(lastOpen)
        lastOpen = float(row["Open"])

    #formats input data into many arrays of len 2 into [[date, yesterdayOpen], [date, yesterdayOpen]]
    dateNums = dateToNum(dates)
    X = [dateNums, yesterdayPrices] #TODO format like [[date, yesterdayOpen], [date, yesterdayOpen]]
    Xcopy = []
    for i in range(0, len(dateNums)):
        Xcopy.append([dateNums[i], yesterdayPrices[i]])
    X = Xcopy
    y = openPrice

    #X = np.array(dateToNum(dates)).reshape(-1, 1) #sklearn expects a 2d array, we need to convert to 2d
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MLPRegressor(
    hidden_layer_sizes=(500, 250, 125),  #number of layers and number of neurons
    activation='relu',         
    solver='adam',              
    max_iter=5000,               
    random_state=42             #makes model the same every run
    )

    print("Training model...")
    model.fit(X_train, y_train) #trains the model
    y_pred = model.predict(X_test)
    print("\nResults:")
    print(f"Accuracy: {round(100 - (metrics.mean_absolute_percentage_error(y_test, y_pred)*100.0), 4)}%")
    print(f"R2: {round(metrics.r2_score(y_test, y_pred), 4)}")
    

#makes each str in dates be represented as number of days after startDate
def dateToNum(dates):
    #30.41 is average number of days in a month
    dateNums = [] #0 represents first date in dataset, all other numbers represent amount of days after the start day
    dateNums.append(0)
    startNum = int(round((getMonth(dates[0]) * 30.41) + getDay(dates[0]) + (getYear(dates[0]) * 365)))
    largestDateNum = 0
    for date in dates:
        if date == dates[0]:
            continue
        thisDateNum = int(round((getMonth(date) * 30.41) + getDay(date) + (getYear(date) * 365))) - startNum
        if thisDateNum < 0:
            print(f"****ERROR: Calculated negative date number {thisDateNum}, ignoring data point**** ")
        else:
            dateNums.append(thisDateNum)
            #print(f"Added {thisDateNum} to dateNums")
        if thisDateNum > largestDateNum:
            largestDateNum = thisDateNum
    return dateNums
        
def getMonth(date):
    return int(date[3:5])

def getDay(date):
    return int(date[0:2])

def getYear(date):
    return int(date[6:])  

def csvToDict(filename):
    dataList = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataList.append(row)
    return dataList

#goes through command line arugments and passes input into main
def parseArgs():
    try:
        if(sys.argv[1].lower() == "search"):
            return ["search", sys.argv[2]]
        else:
            #make sure filename ends in .csv
            filename = sys.argv[1].upper().removesuffix(".CSV")
            filename += ".csv"
            filename = DATASET_PATH + filename
            return [filename]
    except IndexError:
        print(f"Usage: python {FILENAME} {{stock symbol}}\n       python {FILENAME} search {{stock symbol}}")
        return ["error"]
    except FileNotFoundError:
        print(f"Invalid filename, use python {FILENAME} search {{stock symbol}} to find the correct company\n")
        return ["error"]


main()

