import pandas as pd
import math
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
import seaborn as sns
from sklearn import preprocessing

ff_df = pd.read_csv("forestfires.csv")
#applying lambda transform
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))

le = preprocessing.LabelEncoder()
le.fit(ff_df["month"])
ff_df["month"]=le.transform(ff_df["month"])
le.fit(ff_df["day"])
ff_df["day"]=le.transform(ff_df["day"])

pvalues = []
rsquaredvalues = []
colnames=[]

normalized_df=(ff_df.iloc[:,:12]-ff_df.iloc[:,:12].min())/(ff_df.iloc[:,:12].max()-ff_df.iloc[:,:12].min())
normalized_df["area"] = ff_df["area"]

##Linear regression for each variable
model = ols("area ~ X", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["X"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ Y", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["Y"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ month", normalized_df).fit()
print(model.summary())
rsquaredvalues.append(model.rsquared)
for key in list(model.pvalues)[1:]:
    pvalues.append(key)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ day", normalized_df).fit()
print(model.summary())
rsquaredvalues.append(model.rsquared)
for key in list(model.pvalues)[1:]:
    pvalues.append(key)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ FFMC", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["FFMC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ DMC", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["DMC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ DC", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["DC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ ISI", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["ISI"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ temp", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["temp"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ RH", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["RH"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ wind", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["wind"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ rain", normalized_df).fit()
print(model.summary())
pvalues.append(model.pvalues["rain"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

print(pvalues)
print(colnames)


##Graphs on p-values 
y_pos = range(len(colnames))
a = plt.bar(y_pos, pvalues, align='center', alpha=0.5)
plt.xticks(y_pos, colnames,rotation='vertical')
plt.ylabel('p-values')
plt.title('predictors')
plt.show()


colnames2 = []
for column in normalized_df.columns[0:12]:
    colnames2.append(column)

y_pos = range(len(colnames2))
##Graph on r-squared values
plt.bar(y_pos, rsquaredvalues, align='center', alpha=0.5)
plt.xticks(y_pos, colnames2,rotation='vertical')
plt.ylabel('r-squared values')
plt.title('predictors')
plt.show()

#Removing outliers for non -categorical variables
a_df = normalized_df.loc[:, normalized_df.columns != 'month']
new_df = a_df.loc[:,a_df.columns != 'day' ]
new_df = new_df.loc[:,new_df.columns != 'area' ]
new_df = new_df.loc[:,new_df.columns != 'rain' ]
low = .03
high = .97
quant_df = new_df.quantile([low, high])
new_df = new_df.apply(lambda x: x[(x>=quant_df.loc[low,x.name]) & 
                                    (x <= quant_df.loc[high,x.name])], axis=0)
new_df = pd.concat([normalized_df.loc[:,'month'], new_df], axis=1)
new_df = pd.concat([normalized_df.loc[:,'day'], new_df], axis=1)
new_df = pd.concat([normalized_df.loc[:,'area'], new_df], axis=1)
new_df = pd.concat([normalized_df.loc[:,'rain'], new_df], axis=1)
new_df.dropna(inplace=True)
print(new_df)

#Boxplots to show to what extent outliers are removed
colnames = []
for column in normalized_df.columns[0:2]:
    colnames.append(column)
for column in normalized_df.columns[4:12]:
    colnames.append(column)
fig, ax = plt.subplots(nrows=2, ncols = 10, squeeze=True)
j=0
for row in ax:
    i=0
    for col in row:
        
        if j==0:
            col.boxplot(normalized_df[colnames[i]].values)
        if j==1:
            col.boxplot(new_df[colnames[i]].values)
        i=i+1
    j=j+1
plt.show()

#Converting rain to a categorical tackle data imbalance after removing outliers in other features - 0 dominates.

rainlist = np.array(new_df["rain"])
new_rainlist = []
for i in rainlist:
    if i==0:
        new_rainlist.append("No_rain")
    else:
        new_rainlist.append("Yes_rain")
new_df["rain"] = new_rainlist
print(new_df)

#Again running linear regression for data without outliers.

pvalues = []
rsquaredvalues = []
colnames=[]

##Linear regression for each variable
model = ols("area ~ X", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["X"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ Y", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["Y"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ month", new_df).fit()
print(model.summary())
rsquaredvalues.append(model.rsquared)
for key in list(model.pvalues)[1:]:
    pvalues.append(key)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ day", new_df).fit()
print(model.summary())
rsquaredvalues.append(model.rsquared)
for key in list(model.pvalues)[1:]:
    pvalues.append(key)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ FFMC", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["FFMC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ DMC", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["DMC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ DC", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["DC"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ ISI", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["ISI"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ temp", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["temp"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ RH", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["RH"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ wind", new_df).fit()
print(model.summary())
pvalues.append(model.pvalues["wind"])
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

model = ols("area ~ rain", new_df).fit()
print(model.summary())
for key in list(model.pvalues)[1:]:
    pvalues.append(key)
rsquaredvalues.append(model.rsquared)
for key in model.pvalues.keys()[1:]:
    colnames.append(key)

print(pvalues)
print(colnames)


##Graphs on p-values 
y_pos = range(len(colnames))
a = plt.bar(y_pos, pvalues, align='center', alpha=0.5)
plt.xticks(y_pos, colnames,rotation='vertical')
plt.ylabel('p-values')
plt.title('predictors')
plt.show()


colnames2 = []
for column in normalized_df.columns[0:12]:
    colnames2.append(column)

y_pos = range(len(colnames2))
##Graph on r-squared values
plt.bar(y_pos, rsquaredvalues, align='center', alpha=0.5)
plt.xticks(y_pos, colnames2,rotation='vertical')
plt.ylabel('r-squared values')
plt.title('predictors')
plt.show()