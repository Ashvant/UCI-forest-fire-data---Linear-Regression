import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))
g = sns.PairGrid(ff_df.iloc[:,0:8])
g = g.map(plt.scatter)
plt.show()
print(ff_df)