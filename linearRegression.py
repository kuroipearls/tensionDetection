import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics

df = pd.read_csv('dataset_gemastik/try_gemastik10TrainV2.csv', sep=',')
df2 = pd.read_csv('dataset_gemastik/try_gemastik10TestV2.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas
vect = TfidfVectorizer(binary=True)

X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test.values.astype('U'))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train_dtm, y_train)

# Make predictions using the testing set
y_pred_class = regr.predict(X_test_dtm)
y_pred_class = y_pred_class.round()

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
misDF.to_csv("errorAnalysis/error_linregression.csv")

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))
# print(y_pred_class)