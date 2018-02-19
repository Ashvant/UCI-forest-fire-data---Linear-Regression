import pandas as pd
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))

le = preprocessing.LabelEncoder()
le.fit(ff_df["month"])
ff_df["month"]=le.transform(ff_df["month"])
le.fit(ff_df["day"])
ff_df["day"]=le.transform(ff_df["day"])


normalized_df=(ff_df.iloc[:,:12]-ff_df.iloc[:,:12].min())/(ff_df.iloc[:,:12].max()-ff_df.iloc[:,:12].min())
normalized_df["area"] = ff_df["area"]
print(normalized_df)

train, test = train_test_split(normalized_df, test_size=0.1)

scores_training = []
scores_test = []
neighbours = []

for k in range (1,100,2):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neighbours.append(1/k)
    print(k)
    neigh.fit(train.iloc[:,8:12], train["area"])
    result = neigh.predict(train.iloc[:,8:12])
    train["predictedArea"] = result
    rms = np.sqrt(mean_squared_error(train["area"], train["predictedArea"]))
    scores_training.append(rms)
    print(rms)
    result = neigh.predict(test.iloc[:,8:12])
    test["predictedArea"] = result
    rms = np.sqrt(mean_squared_error(test["area"], test["predictedArea"]))
    scores_test.append(rms)
    print(rms)
plt.plot(neighbours,scores_training,label="training mean squared error")
plt.plot(neighbours,scores_test,label="test mean squared error")
plt.xlabel('i/k')
plt.ylabel('mean squared error')
plt.show()