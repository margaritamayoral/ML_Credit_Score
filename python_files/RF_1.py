from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.stats import norm

import plotly.offline as py
#py_init_notebook_mode(conected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter


df = pd.read_csv("../input/german_credit_data_with_risk.csv", index_col=0)

#looking for missings, kind of data and shape:
print(df.info())
#looking for unique values
print(df.nunique())
#Looking the data
print(df.head())

sns.set(style="darkgrid")
ax = sns.countplot(x="Risk", data=df)
plt.savefig('../output/count_risk.png')
#plt.show()
ax2 = plt.figure(2)
ax2 = sns.countplot(x="Risk", hue="Sex", data=df)
plt.savefig('../output/cnt_risk_by_sex.png')
ax3 = plt.figure(3)
ax3 = sns.countplot(x="Risk", hue="Age", data=df)
plt.savefig('../output/cnt_risk_by_age.png')
ax4 = plt.figure(4)
ax4 = sns.catplot(x="Age", hue="Sex", col="Risk", data=df, kind="count", height=4, aspect=.7);
plt.savefig('../output/cnt_risk_sex_age.png');
#####
df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()
df_age = df["Age"].values.tolist()
####First Plot
sns.set_color_codes()
ax5 = plt.figure(5)
ax5 = sns.distplot(df_good, fit=norm, kde=False, color="y")
plt.savefig("../output/dist_good.png")
ax6 = plt.figure(6)
ax6 = sns.distplot(df_bad, fit=norm, kde=False, color="b")
plt.savefig("../output/dist_bad.png")
ax7 = plt.figure(7)
ax7 = sns.distplot(df_age, fit=norm, kde=False, color="r")
plt.savefig("../output/dist_age.png")
#### Second plot
interval = (18, 25, 35, 60, 120)
cats = ['Student', 'Young', 'Adult', 'Senior']
x_age = df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)
y_credit_amount = df["Credit amount"]
print("checking variable")
print(x_age)

sns.set(style="ticks", palette="pastel")
ax8 = plt.figure(8)
ax8 = sns.boxplot(x=x_age, y=y_credit_amount, hue=df["Risk"], palette=["m", "g"], data=df)
plt.savefig("../output/cat_age.png")

##### Third plot
ax9 = plt.figure(9)
ax9 = sns.countplot(x="Housing", hue="Risk", data=df)
plt.savefig("../output/corr_housing_risk.png")

#### Distribution of credit amount by housing
ax10 = plt.figure(10)
plt.figure(figsize=(10, 6))
ax10 = sns.violinplot(x="Housing", y="Credit amount", hue="Risk", side="negative", data=df, palette=["m", "g"])
#ax10 = sns.violinplot(x="Housing", y="Credit amount", inner=None, side="positive", data=df, palette=["g"])
plt.savefig("../output/dist_housing_credit_amount.png")

#### Distribution of credit amount by sex
### Male, Female vs risk count
ax11 = plt.figure(11)
plt.figure(figsize=(10, 6))
ax11a = sns.countplot(x="Sex", hue="Risk", data=df)
plt.savefig("../output/risk_vs_sex.png")

### Credit amount by sex
ax12 = plt.figure(12)
plt.figure(figsize=(10, 6))
ax12 = sns.boxplot(x="Sex", y="Credit amount", hue="Risk", data=df)
plt.savefig("../output/credit_amount_vs_sex.png")

#### Job distribution

### Job vs total credit count and risk credit
ax13 = plt.figure(13)
plt.figure(figsize=(10, 6))
ax13a = sns.countplot(x="Job", data=df, palette=["y"])
ax13b = sns.countplot(x="Job", hue="Risk", data=df)
plt.savefig("../output/job_vs_risk_histo.png")

### Job vs Credit amount
ax14 = plt.figure(14)
plt.figure(figsize=(10, 6))
ax14 = sns.boxplot(x="Job", y="Credit amount", hue="Risk", data=df)
plt.savefig("../output/job_vs_credit_amount_boxplot.png")

ax15 = plt.figure(15)
plt.figure(figsize=(10, 6 ))
ax15 = sns.violinplot(x="Job", y="Credit amount", hue="Risk", data=df)
plt.savefig("../output/job_vs_risk_violinplot.png")

ax16 = plt.figure(16)
fig, ax16 = plt.subplots(figsize(12, 12), nrows=2)
ax16 = sns.boxplot(x="Job", y="Credit amount", hue="Risk", palette="hls", ax16=ax16[0], data=df)
ax16.set_title("Credit Amount by Job", fontsize=15)
ax16.set_xlabel("Job Reference", fontsize=12)
ax16.set_ylabel("Credit Amount", fontsize=12)







### Correlation matrix
## Calculate correlations
corr = df.corr()

#Heatmap
ax11 = plt.figure(11)
ax11 = sns.heatmap(corr)
plt.savefig("../output/general_correlations_risk.png")


