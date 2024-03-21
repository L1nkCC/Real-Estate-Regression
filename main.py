from random import seed
from random import randrange
import random
from csv import reader
from math import exp
from math import sqrt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def natozero(dataset):
    for rowIndex in range(len(dataset)):
        for colIndex in range(len(dataset[rowIndex])):
            if dataset[rowIndex][colIndex] == "NA":
                dataset[rowIndex][colIndex] = "0.0"


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_fs(dataset, column):
    newdataset = []

    fsSet = {"delete"}
    fsSet.clear()
    for row in dataset:
        fsSet.add(row[column])
    fsList = list(fsSet)
    for row in dataset:
        temprow = row[0:column]
        temprow.extend([0.0] * len(fsList))
        temprow.extend(row[column + 1:])
        newdataset.append(temprow)

    for fsIndex in range(len(fsList)):
        for rowIndex in range(len(dataset)):
            if (dataset[rowIndex][column] == fsList[fsIndex]):
                newdataset[rowIndex][column + fsIndex] = 1.0
    return (newdataset, fsList)

def str_column_to_fs_test(dataset, fsListComp,  column):
    newdataset = []
    fsList = []
    for fs in fsListComp:
        if column == fs[1]:
            fsList = fs[0]
            break

    for row in dataset:
        temprow = row[0:column]
        temprow.extend([0.0] * len(fsList))
        temprow.extend(row[column + 1:])
        newdataset.append(temprow)

    for fsIndex in range(len(fsList)):
        for rowIndex in range(len(dataset)):
            try:
                if (dataset[rowIndex][column] == fsList[fsIndex]):
                    newdataset[rowIndex][column + fsIndex] = 1.0
            except:
                print("ERROR")
                print(len(dataset))
                print(len(dataset[0]))
                print(rowIndex)
                print(column)
                exit()
    return newdataset


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        if(value_max == 0 and value_min == 0):
            value_max = 1
        minmax.append([value_min, value_max])
    return minmax


def predict(row, w):
    yhat = w[-1]  # bias is at dimnesion 0
    for i in range(len(row) - 1):
        yhat += w[i] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

def linPredict(row,w):
    yhat = w[-1] # bias is at dimnesion 0
    for i in range(len(row)-1):
        yhat += w[i] * row[i]
    return yhat

def weights_sgd(train, l_rate, n_epoch):
    w = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, w)
            error = row[-1] - yhat
            sum_error += error ** 2
            w[-1] = w[-1] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                w[i] = w[i] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return w

def linWeights_sgd(train, l_rate, n_epoch):
    w = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = linPredict(row, w)
            error = row[-1] - yhat

            sum_error += error**2
            w[-1] = w[-1] + l_rate * error
            for i in range(len(row)-1):
                w[i] = w[i] + l_rate * error * row[i]
    return w

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a cross validation split
def log_evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    weights = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        (predicted, w) = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        weights.append(w)
    return (scores,weights)

def lin_evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    weights = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        (predicted, w) = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
        weights.append(w)
    return (scores, weights)


# Logistic Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    w = weights_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, w)
        yhat = round(yhat)
        predictions.append(yhat)
    return (predictions, w)


# Logistic Regression Algorithm With Stochastic Gradient Descent
def linear_regression(train, test, l_rate, n_epoch):
    predictions = list()
    w = linWeights_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = linPredict(row, w)
        predictions.append(yhat)
    return (predictions, w)



def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) \
            / (minmax[i][1] - minmax[i][0])



#########################################################################################################################
#end of function defintions
#########################################################################################################################


n_folds = 2
l_rate = .1
n_epoch = 5
slinset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
(slinscores, slinw) = lin_evaluate_algorithm(slinset, linear_regression, n_folds, l_rate, n_epoch)

print("Small data set linear:")
for srow in slinset:
    swlistpreds = []
    for w in slinw:
        swlistpreds.append(linPredict(srow,w))
    syhat = sum(swlistpreds)/len(swlistpreds)
    print("Expected=%.3f, Predicted=%.3f" % (srow[-1], syhat))

print()

slogset =[
    [2.7810836,2.550537003,0], [1.465489372,2.362125076,0],
           [3.396561688,4.400293529,0], [1.38807019,1.850220317,0],
           [3.06407232,3.005305973,0], [7.627531214,2.759262235,1],
           [5.332441248,2.088626775,1], [6.922596716,1.77106367,1],
           [8.675418651,-0.242068655,1], [7.673756466,3.508563011,1]
          ]
(slogscores, slogw) = log_evaluate_algorithm(slogset, logistic_regression, n_folds, l_rate, n_epoch)

print("Small data set logistic:")
for srow in slogset:
    swlistpreds = []
    for w in slogw:
        swlistpreds.append(predict(srow,w))
    syhat = sum(swlistpreds)/len(swlistpreds)
    print("Expected=%.3f, Predicted=%.3f [%d]"
          % (srow[-1], syhat, round(syhat)))

print()














# load and prepare housingdata
trainset = load_csv("./train.csv")
testset = load_csv("./test.csv")


linset = trainset[1:]
for rowI in range(len(linset)):
    linset[rowI] = linset[rowI][1:]

testset = testset[1:]
for rowI in range(len(testset)):
    testset[rowI] = testset[rowI][1:]

natozero(linset)
natozero(testset)
olddataset = linset.copy()

i = 0
bound = len(linset[0])
nonfloatindexes = []

while i < bound:
    try:
        str_column_to_float(linset, i)
    except:
        nonfloatindexes.append(i)
    i = i + 1

j = 0
bound2 = len(testset[0])
nonfloatindexes2 = []
while j < bound2:
    try:
        str_column_to_float(testset, j)
    except:
        nonfloatindexes2.append(j)
    j = j + 1



assert(nonfloatindexes2 == nonfloatindexes)
nonfloatindexes.reverse()

testfsListComp = []
for nfi in nonfloatindexes:

    (linset, linfsList) = str_column_to_fs(linset, nfi)
    testfsListComp.append([linfsList, nfi])
    testset = str_column_to_fs_test(testset, testfsListComp, nfi)

logset = []
for rowI in range(len(linset)):
    tempRow = []
    for colI in range(len(linset[rowI])):
        tempRow.append(linset[rowI][colI])
    logset.append(tempRow)

for row in logset:
    if row[-1]>180000:
        row[-1] = 1
    else:
        row[-1] = 0



# normalize
templinset = []
for rowI in range(len(linset)):
    temprow = []
    for colI in range(len(linset[rowI])-1):
        temprow.append(linset[rowI][colI])
    templinset.append(temprow)
minmax1 = dataset_minmax(templinset)
normalize_dataset(templinset, minmax1)

for rowI in range(len(linset)):
    for colI in range(len(linset[rowI])):
        if colI != len(linset[rowI])-1:
            linset[rowI][colI] = templinset[rowI][colI]


minmax = dataset_minmax(logset)
normalize_dataset(logset, minmax)

minmax = dataset_minmax(testset)
normalize_dataset(testset, minmax)

# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 50

(logScores, logw) = log_evaluate_algorithm(logset, logistic_regression,
                            n_folds, l_rate, n_epoch)
print('Logistic Scores: %s' % logScores)
print('Logistic Mean Accuracy: %.3f%%'
      % (sum(logScores) / float(len(logScores))))
print('logw : %s' % logw)
print('\n')




(linScores, linw) = lin_evaluate_algorithm(linset, linear_regression,
                            n_folds, l_rate, n_epoch)
print('Linear Scores: %s' % linScores)
print('Linear Mean RMSE: %.3f'
      % (sum(linScores) / float(len(linScores))))
print('linw : %s' % linw)

print()
for rowI in range(len(testset)):
    preds = []
    for w in linw:
        preds.append(linPredict(testset[rowI],w))
    print('Test.cvs Item %d: %.2f    (linear result)' % (rowI, sum(preds)/len(preds)))
    logpreds = []
    for w in logw:
        logpreds.append(predict(testset[rowI],w))
    if sum(logpreds)/len(logpreds) > .5:
        print('     likely over 180000 [%.3f] (logistic result)' % (sum(logpreds)/len(logpreds)))
    else:
        print('     likely under 180000 [%.3f] (logistic result)' % (sum(logpreds)/len(logpreds)))




