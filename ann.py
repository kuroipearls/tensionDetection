import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('datasetFinal/dataset3class_V4fix.csv', sep=',')

X_train = df.text
y_train = df.is_kelas

vect = TfidfVectorizer(binary=True)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))

# parameters = {'hidden_layer_sizes':{(7,2),(5,2)}}
parameters = dict(hidden_layer_sizes = [(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10)])

clf = MLPClassifier(solver='adam',alpha=1e-6, max_iter=500, learning_rate_init=0.0001)
print(clf.get_params().keys())

grid = GridSearchCV(clf, parameters)
grid.fit(X_train_dtm,y_train)
y_pred_class = cross_val_predict(clf, X_train_dtm, y_train, cv=5)
# print('Best Solver:',grid.best_estimator_.solver)
print('Best hidden layer sizes:',grid.best_estimator_.hidden_layer_sizes)
print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_class))
print(metrics.confusion_matrix(y_train, y_pred_class))