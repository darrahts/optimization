import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cross_validation

Data = pd.read_csv("spambase.csv")

Data = Data.sample(frac=1)

Xold = Data.drop(["Y"],axis=1)
Y = pd.DataFrame(Data,columns=["Y"]).values


norm = preprocessing.MinMaxScaler()
X = norm.fit_transform(Xold)


pca = PCA(n_components=30)
X = pca.fit_transform(X)

Var = pca.explained_variance_ratio_

Exp_Var = 0

for i in Var:
    Exp_Var += i
'''
print("\n")
print(Exp_Var)
'''
kfold = 10

SVMClass = svm.SVC()


Count1 = 1
Aa1 = 0

Cnt1 = len(X)

kf = cross_validation.KFold(Cnt1, n_folds=kfold)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = SVMClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("Accuracy for SVM W/O GA: %f" % (Aa1/kfold))



SVMClass = svm.SVC(kernel='rbf', C=993, gamma=0.13)

Count1 = 1
Aa1 = 0

Cnt1 = len(X)

kf = cross_validation.KFold(Cnt1, n_folds=kfold)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = SVMClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("Accuracy for SVM W/ GA: %f" % (Aa1/kfold))





