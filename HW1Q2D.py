import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn import preprocessing

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))

le = preprocessing.LabelEncoder()
le.fit(ff_df["month"])
ff_df["month"]=le.transform(ff_df["month"])
le.fit(ff_df["day"])
ff_df["day"]=le.transform(ff_df["day"])

normalized_df=(ff_df.iloc[:,:12]-ff_df.iloc[:,:12].min())/(ff_df.iloc[:,:12].max()-ff_df.iloc[:,:12].min())
normalized_df["area"] = ff_df["area"]

model = ols("area ~ X+Y+month+day+FFMC+DMC+DC+ISI+temp+RH+wind+rain", normalized_df).fit()
print(model.summary())

pvalues = []
keys = []
intercept=[]

for key in model.pvalues.keys()[1:]:
    intercept.append(model.params[key])
    keys.append(key)
    pvalues.append(model.pvalues[key])

y_pos = range(len(keys))
a = plt.bar(y_pos, intercept, align='center', alpha=0.5)
plt.xticks(y_pos, keys,rotation='vertical')
plt.ylabel('intercept values')
plt.title('predictors')
plt.show()

y_pos = range(len(keys))
a = plt.bar(y_pos, pvalues, align='center', alpha=0.5)
plt.xticks(y_pos, keys,rotation='vertical')
plt.ylabel('p-values')
plt.title('predictors')
plt.show()