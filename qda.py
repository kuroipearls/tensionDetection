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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

df = pd.read_csv('datasetFinal/dataset2class_V4fix.csv', sep=',')

X_train = df.text
y_train = df.is_kelas

# vect = TfidfVectorizer(binary=True)
# X_train_dtm = vect.fit_transform(X_train.values.astype('U'))

clf = make_pipeline(
     TfidfVectorizer(), 
     FunctionTransformer(lambda x: x.todense(), accept_sparse=True), 
     LinearDiscriminantAnalysis()
)

clf.fit(X_train,y_train)

y_pred_class = cross_val_predict(clf, X_train, y_train, cv=5)
# print('Best Solver:',grid.best_estimator_.solver)

print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_class))
print(metrics.confusion_matrix(y_train, y_pred_class))