import numpy as np
import pandas as pd
import random as rd
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

# Loading the data, shuffling and preprocessing it

Data = pd.read_csv("C:/Users/Dana Bani-Hani/Desktop/My Files/My Courses/Machine Learning Optimization Using Genetic Algorithm/Dataset.csv")

Data = Data.sample(frac=1)

X1 = pd.DataFrame(Data,columns=['X1','X2','X3','X4','X5','X6','X7','X8'])

Y = pd.DataFrame(Data,columns=['Y1']).values


Xbef = pd.get_dummies(X1,columns=['X6','X8'])


min_max_scalar = preprocessing.MinMaxScaler()

X = min_max_scalar.fit_transform(Xbef)

Cnt1 = len(X)
print()
print("# of Observations:",Cnt1)


kfold = 10

MLPClass = MLPRegressor()

Count1 = 1
Aa1 = 0

Cnt1 = len(X)

kf = cross_validation.KFold(Cnt1, n_folds=kfold)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = MLPClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("R2 for MLP W/O GA: %f" % (Aa1/kfold))




MLPClass = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(8,8,8),
                           learning_rate_init=0.017,momentum=0.094)

Count1 = 1
Aa1 = 0

Cnt1 = len(X)

kf = cross_validation.KFold(Cnt1, n_folds=kfold)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = MLPClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("R2 for MLP W/ GA: %f" % (Aa1/kfold))

















