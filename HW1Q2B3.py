import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

ff_df = pd.read_csv("forestfires.csv")
ff_df['area']=ff_df['area'].apply(lambda x:math.log1p(x))
g = sns.PairGrid(ff_df, vars=["X", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["Y", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["month", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["day", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["FFMC", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["DMC", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["DC", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["ISI", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["temp", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["RH", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["wind", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
g = sns.PairGrid(ff_df, vars=["rain", "area"],size=3)
g = g.map(plt.scatter)
plt.show()
print(ff_df)