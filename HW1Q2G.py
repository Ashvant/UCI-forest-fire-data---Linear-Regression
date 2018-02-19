import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.feature_selection import RFE
from sklearn import preprocessing

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))
intercept1=[]
pvalues1 =[]
keys1=[]
intercept2=[]
pvalues2 =[]
keys2=[]
intercept3=[]
pvalues3 =[]
keys3=[]

le = preprocessing.LabelEncoder()
le.fit(ff_df["month"])
ff_df["month"]=le.transform(ff_df["month"])
le.fit(ff_df["day"])
ff_df["day"]=le.transform(ff_df["day"])

normalized_df=(ff_df.iloc[:,:12]-ff_df.iloc[:,:12].min())/(ff_df.iloc[:,:12].max()-ff_df.iloc[:,:12].min())
normalized_df["area"] = ff_df["area"]

model = ols("area ~ X+Y+month+day+FFMC+DMC+DC+ISI+temp+RH+wind+rain+X:Y+X:month+X:day+X:FFMC+X:DMC+X:DC+X:ISI+X:temp+X:RH+X:wind+X:rain+Y:month+Y:day+Y:FFMC+Y:DMC+Y:DC+Y:ISI+Y:temp+Y:RH+Y:wind+Y:rain+month:day+month:FFMC+month:DMC+month:DC+month:ISI+month:temp+month:RH+month:wind+month:rain+day:FFMC+day:DMC+day:DC+day:ISI+day:temp+day:RH+day:wind+day:rain+FFMC:DMC+FFMC:DC+FFMC:ISI+FFMC:temp+FFMC:RH+FFMC:wind+FFMC:rain+DMC:DC+DMC:ISI+DMC:temp+DMC:RH+DMC:wind+DMC:rain+DC:ISI+DC:temp+DC:RH+DC:wind+DC:rain+ISI:temp+ISI:RH+ISI:wind+ISI:rain+temp:RH+temp:wind+temp:rain+RH:wind+RH:rain+wind:rain", normalized_df).fit()
print(model.summary())

for key in model.pvalues.keys()[1:40]:
    intercept1.append(model.params[key])
    pvalues1.append(model.pvalues[key])
    keys1.append(key)
for key in model.pvalues.keys()[40: ]:
    intercept2.append(model.params[key])
    pvalues2.append(model.pvalues[key])
    keys2.append(key)


##Graphs on p-values 
y_pos = range(len(keys1))
a = plt.bar(y_pos, pvalues1,align='center', alpha=0.5)
plt.xticks(y_pos, keys1,size=8,rotation=90)
plt.ylabel('p-values')
plt.title('predictors')
plt.show()

##Graphs on p-values 
y_pos = range(len(keys2))
a = plt.bar(y_pos, pvalues2, align='center', alpha=0.5)
plt.xticks(y_pos, keys2,rotation='vertical')
plt.ylabel('p-values')
plt.title('predictors')
plt.show()

