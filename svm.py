import pandas as pd
import numpy as np
import time
import csv
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

df = pd.read_csv('dataset_gemastik/try_gemastik10TrainV2.csv', sep=',')
df2 = pd.read_csv('dataset_gemastik/try_gemastik10TestV2.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas

# vect = CountVectorizer(binary=True)
vect = TfidfVectorizer(binary=True)

X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test.values.astype('U'))

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas

parameters = {'kernel':('linear', 'rbf'), 'C':[1,5,10,15,25,50,75,100,150,200,250], 'gamma': 
[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5]}

svr = svm.SVC()
grid = GridSearchCV(svr, parameters)
grid.fit(X_train_dtm,y_train)
y_pred_class = grid.predict(X_test_dtm)
misDataTeks = [teks
          for teks, truth, prediction in
          zip(X_test, y_test, y_pred_class)
          if truth != prediction]
misDataTruth = [truth
          for teks, truth, prediction in
          zip(X_test, y_test, y_pred_class)
          if truth != prediction]
misDataPred = [prediction
          for teks, truth, prediction in
          zip(X_test, y_test, y_pred_class)
          if truth != prediction]

misDF = pd.DataFrame({'teks':misDataTeks, 'actual':misDataTruth, 'prediction':misDataPred})

print('Best C:',grid.best_estimator_.C)
print('Best Kernel:',grid.best_estimator_.kernel)
print('Best Gamma:',grid.best_estimator_.gamma)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))

misDF.to_csv("errorAnalysis/error_svm.csv")