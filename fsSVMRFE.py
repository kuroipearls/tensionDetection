import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('datasetFinal/dataset3class_V5fixx.csv', sep=',')

X_train = df.text
y_train = df.is_kelas

vect = TfidfVectorizer(binary=True)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))

svr = LinearSVC(C=5)

# rfe = RFECV(estimator = svr, cv = 10, scoring="accuracy")
# rfe = rfe.fit(X_train_dtm, y_train)
# print("Optimal number of features : %d" % rfe.n_features_)

# y_pred_class = cross_val_predict(rfe, X_train_dtm, y_train, cv=10)

# print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_class))
# print(metrics.confusion_matrix(y_train, y_pred_class))

rfe = RFE(estimator = svr, n_features_to_select=298, step=1)
rfe = rfe.fit(X_train_dtm, y_train)
y_pred_class = cross_val_predict(rfe, X_train_dtm, y_train, cv=10)
print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_class))
print(metrics.confusion_matrix(y_train, y_pred_class))

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("CV Score")
# plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
# plt.savefig('rfe.png', bbox_inches='tight')
# plt.show()