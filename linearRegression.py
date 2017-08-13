import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('dataset_gemastik/dataset_jokowiV1fixclean.csv', sep=',')
df2 = pd.read_csv('dataset_gemastik/try_gemastik10TestV2.csv', sep=',')

### USE IT WITH SEPARATE TRAINING SET & TESTING SET ###
# X_train = df.text
# y_train = df.is_kelas
# X_test = df2.text
# y_test = df2.is_kelas
# vect = TfidfVectorizer(binary=True)

# X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# # print(X_train_dtm)
# X_test_dtm = vect.transform(X_test.values.astype('U'))

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(X_train_dtm, y_train)

# # Make predictions using the testing set
# y_pred_class = regr.predict(X_test_dtm)
# y_pred_class = y_pred_class.round()

# misDataTeks = [teks
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDataTruth = [truth
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDataPred = [prediction
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDF = pd.DataFrame({'teks':misDataTeks, 'actual':misDataTruth, 'prediction':misDataPred})
# misDF.to_csv("errorAnalysis/error_linregression.csv")

# print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_class))
# print(metrics.confusion_matrix(y_test, y_pred_class))
# # print(y_pred_class)
### END - USE IT WITH SEPARATE TRAINING AND TESTING SET ###

### USE IT FOR CROSS VALIDATION ###
X_train = df.text
y_train = df.is_kelas

vect = TfidfVectorizer(binary=True)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))

clf = linear_model.LinearRegression()
# scores = cross_val_score(clf, X_train_dtm, y_train, cv=5)
y_pred_class = cross_val_predict(clf, X_train_dtm, y_train, cv=5)
y_pred_class = y_pred_class.round()

misDataTeks = [teks
          for teks, truth, prediction in
          zip(X_train, y_train, y_pred_class)
          if truth != prediction]
misDataTruth = [truth
          for teks, truth, prediction in
          zip(X_train, y_train, y_pred_class)
          if truth != prediction]
misDataPred = [prediction
          for teks, truth, prediction in
          zip(X_train, y_train, y_pred_class)
          if truth != prediction]

misDF = pd.DataFrame({'teks':misDataTeks, 'actual':misDataTruth, 'prediction':misDataPred})
print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_class))
print(metrics.confusion_matrix(y_train, y_pred_class))
misDF.to_csv("errorAnalysis/error_linregressionJKW.csv")

# print(scores)
### END - USE IT FOR CROSS VALIDATION ### 