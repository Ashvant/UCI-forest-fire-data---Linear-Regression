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
#print(list(model.params)[1:])

multiple = dict()
for key in model.params.keys()[1:]:
    multiple[key] = model.params[key] 
#print(multiple)

single=dict()
model = ols("area ~ X", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ Y", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ month", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ day", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ FFMC", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ DMC", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ DC", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ ISI", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ temp", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ RH", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ wind", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]

model = ols("area ~ rain", normalized_df).fit()
for key in model.params.keys()[1:]:
    single[key] = model.params[key]
#print(single)

singleList = []
multipleList = []
keyList = []
for key in single.keys():
    singleList.append(single[key])
    multipleList.append(multiple[key])
    keyList.append(key)

df = pd.DataFrame()
df["univariate"] = singleList
df["multivariate"] = multipleList
df["keys"] = keyList
print(df)
sns.pairplot(x_vars=["univariate"], y_vars=["multivariate"], data=df, hue="keys", size=5)
plt.show()