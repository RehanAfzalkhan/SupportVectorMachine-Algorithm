import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
# from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
heart_df=pd.read_csv('dataset/heart.csv')
heart_df=pd.get_dummies(heart_df,drop_first=True,dtype=float)
X = heart_df.drop("HeartDisease",axis=1).values
y = heart_df["HeartDisease"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
scalar=StandardScaler()
X_train_scalad=scalar.fit_transform(X_train)
X_test_scalad=scalar.transform(X_test)
svm_model=SVC()
svm_model.fit(X_train_scalad,y_train)
y_pred=svm_model.predict(X_test_scalad)
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy score is:",accuracy_score)
