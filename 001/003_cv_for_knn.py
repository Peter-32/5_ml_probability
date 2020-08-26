import os
import random
import pandas as pd
from sklearn.model_selection import KFold,cross_validate
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_dir = os.path.abspath(os.path.dirname(__file__))

df = pd.read_csv(f"{base_dir}/data/train.csv")
train, test = train_test_split(df, test_size=0.40)
X_train = train.drop(['label'], axis='columns')
y_train = train[['label']]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


kf=KFold(n_splits)
            model_score=cross_validate(pipe,data,target,scoring=metrics,cv=kf)

print("The accuracy is {:.1%} for Cross Validation".format(accuracy))

# KNN is predict all test set rows slowly
