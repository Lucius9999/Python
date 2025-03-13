import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %matplotlib inline

df = pd.read_csv('C:/Users/LPora/OneDrive - DXC Production/Trainings/NVIDIA/Data Set/Advertising.csv')
# print(df)

# print(h1)
# df.head(2)
df.drop(columns = "Unnamed: 0",inplace=True)
h1 = df.head()
# print(h1)
X = df.drop(columns='Sales')
y = df['Sales']
# df.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 4, test_size =0.3)
# X_train.shape,X_test.shape

lr = LinearRegression()
lr.fit(X_train,y_train )
lr.coef_
print(lr.coef_)
lr.intercept_
print(lr.intercept_)

y_pred_test = lr.predict(X_test)
X_test[:5]
print(X_test[:5])
print(y_pred_test[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f"MSE is {mean_squared_error(y_test, y_pred_test)}")
print(f"RMSE is {mean_squared_error(y_test, y_pred_test, squared = False)}")
print(f"MAE is {mean_absolute_error(y_test,y_pred_test)}")

#R2 score
# How much variance can be explained by the given features
print(f"R2-score is {r2_score(y_test,y_pred_test)}" )