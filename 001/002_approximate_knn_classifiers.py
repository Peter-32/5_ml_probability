from pyflann import *

import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_dir = os.path.abspath(os.path.dirname(__file__))

df = pd.read_csv("{}/data/train.csv".format(base_dir))
train, test = train_test_split(df, test_size=0.40)
X_train = train.drop(['label'], axis='columns')
column_names = X_train.columns.tolist()
random.shuffle(column_names)
X_train = X_train[column_names]
y_train = train[['label']]
X_test = test.iloc[0:1000].drop(['label'], axis='columns')
X_test = X_test[column_names]
y_test = test.iloc[0:1000][['label']]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

flan = FLANN()
result, dists = flann.nn(X_train, X_test, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
print result
print dists
