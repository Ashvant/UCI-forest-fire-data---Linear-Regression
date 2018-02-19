import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))
describe = ff_df.describe()
#print(describe.loc["mean"])
table = pd.DataFrame()
table["Mean"] = describe.loc["mean"] 
table["Median"] = ff_df.median()
#table["Range"] = describe.loc["max"] + "-" + describe.loc["min"]
table["Range"] = describe.loc["max"] - describe.loc["min"]
table["FistQuartile"] = describe.loc["25%"]
table["ThirdQuartile"] = describe.loc["75%"]
table["Interquartile"] = describe.loc["75%"]-describe.loc["25%"]
print(final)