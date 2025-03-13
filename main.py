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
print(h1)
