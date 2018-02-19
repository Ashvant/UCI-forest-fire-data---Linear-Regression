import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))
formulaList = ["X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain","X:Y","X:FFMC","X:DMC","X:DC","X:ISI","X:temp","X:RH","X:wind","X:rain","Y:FFMC","Y:DMC","Y:DC","Y:ISI","Y:temp","Y:RH","Y:wind","Y:rain","FFMC:DMC","FFMC:DC","FFMC:ISI","FFMC:temp","FFMC:RH","FFMC:wind","FFMC:rain","DMC:DC","DMC:ISI","DMC:temp","DMC:RH","DMC:wind","DMC:rain","DC:ISI","DC:temp","DC:RH","DC:wind","DC:rain","ISI:temp","ISI:RH","ISI:wind","ISI:rain","temp:RH","temp:wind","temp:rain","RH:wind","RH:rain","wind:rain","month:day","month:FFMC","month:DMC","month:DC","month:ISI","month:temp","month:wind","month:rain","month:RH","day:FFMC","day:DMC","day:DC","day:ISI","day:temp","day:RH","day:wind","day:rain"]

le = preprocessing.LabelEncoder()
le.fit(ff_df["month"])
ff_df["month"]=le.transform(ff_df["month"])
le.fit(ff_df["day"])
ff_df["day"]=le.transform(ff_df["day"])

normalized_df=(ff_df.iloc[:,:12]-ff_df.iloc[:,:12].min())/(ff_df.iloc[:,:12].max()-ff_df.iloc[:,:12].min())
normalized_df["area"] = ff_df["area"]

for column in list(normalized_df.columns[:-1]):
        formulaList.append("I("+column+" ** 2.0)")
        formulaList.append("I("+column+" ** 3.0)")
#print(formulaList)
bestFormula = None
bestRsquared = 100
train, test = train_test_split(normalized_df, test_size=0.3)
while (len(formulaList)>2
"0):
    formula_now = ""
    for formula in formulaList:
        formula_now+="+"+formula
    model = ols("area ~"+formula_now, train).fit()
    op  = model.predict(test)
    test["predictedArea"] = op
    rms = np.sqrt(mean_squared_error(test["area"], test["predictedArea"]))
    pvalues = pd.DataFrame()
    pvalues = model.pvalues[1:].sort_values(ascending=False)
#    print(model.pvalues[1:].sort_values(ascending=False))
#    print(pvalues.keys()[0])
    formulaList.remove(pvalues.keys()[0])
    if(bestRsquared > rms):
        bestFormula = formula_now
        bestRsquared = rms
print(bestFormula)
print(bestRsquared)

model = ols("area ~"+bestFormula, train).fit()
print(model.summary())
op  = model.predict(test)
#print(op)
test["predictedArea"] = op
print(test)
rms = np.sqrt(mean_squared_error(test["area"], test["predictedArea"]))
print(rms)