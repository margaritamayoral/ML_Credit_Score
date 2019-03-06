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

###############################
from sklearn.model_selection import train_test_split, KFold, cross_val_score  ### used to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score   ### to evaluate the model

from sklearn.model_selection import GridSearchCV

#### Algorithms models to be compared

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#from xgboost import XGBClassifier

#########################################
#### Model 2
from sklearn.utils import resample
from sklearn.metrics import roc_curve

#######################################
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest



#######################################

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
plt.savefig('../output/cnt_risk_sex_age.png')
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
fig, ax = plt.subplots(figsize=(12, 12), nrows=2)
ax16 = sns.boxplot(x="Job", y="Credit amount", hue="Risk", palette="hls", ax=ax[0], data=df)
ax16.set_title("Credit Amount by Job", fontsize=15)
ax16.set_xlabel("Job Reference", fontsize=12)
ax16.set_ylabel("Credit Amount", fontsize=12)

ax17 = sns.violinplot(x="Job", y="Age", hue="Risk", split=True, pallete="hls", ax=ax[1], data=df)
ax17.set_title("Job class vs Age", fontsize=15)
ax17.set_xlabel("Job class", fontsize=12)
ax17.set_ylabel("Age", fontsize=12)
plt.subplots_adjust(hspace=0.4, top=0.9)
plt.savefig("../output/job_vs_age_credit_amount_risk.png")
#plt.show()

#Visualizing the distribution of credit amount

#Histogram:
x1 = np.log(df.loc[df["Risk"] == 'good']['Credit amount'])
x2 = np.log(df.loc[df["Risk"] == 'bad']['Credit amount'])
#histo = [x1, x2]
#group_labels = ["Good credit", "Bad credit"]

ax18 = plt.figure(18)
#ax18 = sns.distplot(x1)
ax18 = sns.distplot(x1, label="Good credit",  color="r")
ax18 = sns.distplot(x2, label="Bad credit", color="b")
ax18.set_title("Credit amount distribution", fontsize=15)
ax18.set_xlabel("Good Credit (red), Bad Credit (blue)", fontsize=12)
ax18.set_ylabel("Credit amount", fontsize=12)
plt.savefig("../output/Credit_amount_distribution.png")

### Distribution of saving accounts by risk
print("Description of distribution of Saving Accounts by Risk")
print(pd.crosstab(df["Saving accounts"], df.Risk))

ax19 = plt.figure(19)
fig, ax = plt.subplots(3, 1, figsize=(12, 12))
ax19 = sns.countplot(x="Saving accounts", hue="Risk", palette="hls", ax=ax[0], data=df)
ax19.set_title("Count saving accounts", fontsize=15)
ax19.set_xlabel("Saving account class", fontsize=12)
ax19.set_ylabel("Count", fontsize=12)

ax20 = sns.boxplot(x="Saving accounts", y="Credit amount", hue="Risk", palette="hls", ax=ax[1], data=df)
ax20.set_title("Credit amount by Savings accounts", fontsize=15)
ax20.set_xlabel("Saving account class", fontsize=12)
ax20.set_ylabel("Credit amount", fontsize=12)

ax21 = sns.boxplot(x="Saving accounts", y="Age", hue="Risk", palette="hls", data=df)
ax21.set_title("Savings accounts and Age relation", fontsize=15)
ax21.set_xlabel("Saving account class", fontsize=12)
ax21.set_ylabel("Age", fontsize=12)

plt.subplots_adjust(hspace=0.4, top=0.9)
plt.savefig("../output/Saving_account_relations_distribution.png")

#### Distribution of purposes

print("Description of distribution of purposes")
print(pd.crosstab(df["Purpose"], df.Risk))

ax22 = plt.figure(22)
fig, ax = plt.subplots(3, 1, figsize=(12, 12))
ax22 = sns.countplot(x="Purpose", hue="Risk", palette="hls", ax=ax[0], data=df)
ax22.set_title("Count Purposes", fontsize=15)
ax22.set_xlabel("Purpose class", fontsize=12)
ax22.set_ylabel("Count", fontsize=12)

ax23 = sns.boxplot(x="Purpose", y="Credit amount", hue="Risk", palette="hls", ax=ax[1], data=df)
ax23.set_title("Credit amount by Purpose", fontsize=15)
ax23.set_xlabel("Purpose class", fontsize=12)
ax23.set_ylabel("Credit amount", fontsize=12)

ax24 = sns.boxplot(x="Purpose", y="Age", hue="Risk", palette="hls", data=df)
ax24.set_title("Purpose and Age relation", fontsize=15)
ax24.set_xlabel("Purpose class", fontsize=12)
ax24.set_ylabel("Age", fontsize=12)

plt.subplots_adjust(hspace=0.4, top=0.9)
plt.savefig("../output/purpose_vs_age_credit_amount_risk.png")

## Duration of the loans distribution and density
df_good = df.loc[df["Risk"] == 'good']
df_bad = df.loc[df["Risk"] == 'bad']

ax25 = plt.figure(25)
fig, ax = plt.subplots(3, 1, figsize=(12, 14))
ax25 = sns.countplot(x="Duration", hue="Risk", palette="hls", ax=ax[0], data=df)
ax25.set_title("Duration Count", fontsize=20)
ax25.set_xlabel("Duration Distribution", fontsize=12)
ax25.set_ylabel("Count", fontsize=12)

ax26 = sns.pointplot(x="Duration", y="Credit amount", hue="Risk", palette="hls", ax=ax[1], data=df)
ax26.set_title('Credit amount distribution by duration', fontsize=20)
ax26.set_xlabel("Duration", fontsize=12)
ax26.set_ylabel("Credit amount", fontsize=12)


ax27 = sns.distplot(df_good["Duration"], color="g")
ax27 = sns.distplot(df_bad["Duration"], color="r")
ax27.set_title("Duration of credit vs good or bad behaviour", fontsize=15)
ax27.set_xlabel("Duration", fontsize=12)
ax27.set_ylabel("Frequency", fontsize=12)
plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9)
plt.savefig("../output/Credit_Duration_vs_Credit_Behaviour.png")

## Checking Account variable

### Checking the distribution

ax28 = plt.figure(28)
fig, ax = plt.subplots(3, 1, figsize=(12, 14))
ax28 = sns.countplot(x="Checking account", hue="Risk", palette="hls", ax=ax[0], data=df)
ax28.set_title("Checking account vs Risk", fontsize=15)
ax28.set_xlabel("Checking account class", fontsize=12)
ax28.set_ylabel("Count", fontsize=12)

ax29 = sns.boxplot(x="Checking account", y = "Credit amount", hue="Risk", palette="hls", data=df, ax=ax[1])
ax29.set_title("Checking account distribution in function of Credit Amount", fontsize=15)
ax29.set_xlabel("Checking account", fontsize=12)
ax29.set_ylabel("Credit amount", fontsize=12)

ax30 = sns.violinplot(x="Checking account", y="Age", hue="Risk", palette="hls", data=df, ax=ax[2])
ax30.set_title("Checking account vs Age", fontsize=15)
ax30.set_xlabel("Checking account class", fontsize=12)
ax30.set_ylabel("Age", fontsize=12)

plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9)
plt.savefig("../output/Checking_account_vs_Risk.png")

### Crosstab to explain more insights in the data




print(pd.crosstab(df.Sex, df.Job))

ax31 = plt.figure(31)
ax31 = sns.violinplot(x="Housing", y="Age", hue="Risk", palette="hls", data=df)
ax31.set_title("Exploring Housing vs Age", fontsize=15)
ax31.set_xlabel("Housing", fontsize=12)
ax31.set_ylabel("Age", fontsize=12)
plt.savefig("../output/Housing_Age.png")

print(pd.crosstab(df["Checking account"], df.Sex))


date_int = ["Purpose", "Sex"]
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df[date_int[0]], df[date_int[1]]).style.background_gradient(cmap=cm)
#can't be printed in run console

#Looking at the total of values in each one of the features
print("Purpose:", df.Purpose.unique())
print("Sex:", df.Sex.unique())
print("Housing:", df.Housing.unique())
print("Saving accounts:", df['Saving accounts'].unique())
print("Risk:", df['Risk'].unique())
print("Checking account:", df['Checking account'].unique())
print("Aget_cat:", df['Age_cat'].unique())


#### Feature Engineering, creating variable Dummies of the values #######

def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#### Transforming the data into Dummy variables

df['Saving_accounts'] = df['Saving accounts'].fillna(0)
df['Checking account'] = df['Checking account'].fillna(0)
df.loc[df['Saving accounts'] == 'quite rich', 'Saving accounts'] = 'quite_rich'


#Purpose to Dummy variables

df = df.merge(pd.get_dummies(df.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#print(df.head())

#Sex feature in dummies

df = df.merge(pd.get_dummies(df.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
#print(df.head())

### Housing get dummies

df = df.merge(pd.get_dummies(df.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
#print(df.head())

### Saving accounts feature in dummies

df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
print("Saving accounts checked:")
print(df.head())

### Risk dummies

df = df.merge(pd.get_dummies(df.Risk, prefix="Risk"), left_index=True, right_index=True)

### Checking account dummies

df = df.merge(pd.get_dummies(df["Checking account"], drop_first=True, prefix="Check"), left_index=True, right_index=True)
#print(df.head())

### Age_cat dummies

df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix="Age_cat"), left_index=True, right_index=True)
#print(df.head())
#df.info()

### Excluding the old columns

del df["Saving accounts"]
del df["Saving_accounts"]
del df["Checking account"]
del df["Purpose"]
del df["Sex"]
del df["Housing"]
del df["Age_cat"]
del df["Risk"]
del df["Risk_good"]

print("information about the last df:")
print(df.head())
print(df.info())
print(df.describe())


### Correlation

### Looking the correlation

### Correlation matrix
## Calculate correlations
corr = df.corr()

#Heatmap
ax32 = plt.figure(32)
ax32 = sns.heatmap(corr)
plt.savefig("../output/general_correlations_risk.png")

ax33 = plt.figure(33, figsize=(14, 12))
ax33 = sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor="white", annot=True)
plt.savefig("../output/features_correl_heatmap.png")

### Preprocessing
df['Credit amount'] = np.log(df['Credit amount'])
print(np.log(df['Credit amount']).head())


### Creating X and y variables

X = df.drop('Risk_bad', 1).values
y = df["Risk_bad"].values

# Splitting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


####################################################
#print("EXIST STRING?")

#validation = []
#for x in X_test:
#    s = x.dtype.type is np.string_
#    validation.append(s)
#    print(validation)

######################################################
#to feed the random state

seed = 7

#prepare models

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', GradientBoostingClassifier()))

## evaluating each model

results = []
names = []
scoring = 'recall'

for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        print(kfold)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        print(cv_results)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


ax34 = plt.figure(figsize=(11, 16))
ax34.suptitle('Algorithms Comparison')
ax34 = ax34.add_subplot(111)
plt.boxplot(results)
ax34.set_xticklabels(names)
plt.savefig("../output/Algorithms_Comparison.png")

##### Model 1 using Random Forest to predict credit score

#Setting the hyperparameters:

param_grid = {"max_depth": [3, 5, 7, 10, None], "n_estimators": [3, 5, 10, 25, 50, 150],
              "max_features": [4, 7, 15, 20]}

#Creating the classifier:
model = RandomForestClassifier(random_state=2)

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)
#Training with the best parameters
rf.fit(X_train, y_train)

# Testing the model
#Predicting using the model
y_pred = rf.predict(X_test)

#Verifying:

print(accuracy_score(y_test, y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))

#### Model 2

#Classificador logreg

GNB = GaussianNB()

#Fitting with train data

model2 = GNB.fit(X_train, y_train)

#Printing the training score
print("Training score data:")
print(model2.score(X_train, y_train))

y_pred2 = model2.predict(X_test)

print(accuracy_score(y_test, y_pred2))
print("\n")
print(confusion_matrix(y_test, y_pred2))
print("\n")
print(classification_report(y_test, y_pred2))


#### ROC curve

# Predicting proba
y_pred_prob = model2.predict_proba(X_test)[:, 1]
# Generate Roc curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Plot ROC curve
ax35 = plt.figure(figsize=(11, 16))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.savefig("../output/ROC_curve.png")

features = []
features.append(('pca', PCA(n_components=2)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
#create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', GaussianNB()))
model3 = Pipeline(estimators)
#evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

print(accuracy_score(y_test, y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))


### Implementing a pipeline model

## Setting the hyperparameters
#param_test1 = {
#    'max_depth': [3, 5, 6, 10]
#    'min_child_weight':[3, 5, 10]
#    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
#    'subsample': [i/100.0 for i in range(75, 90, 5)],
#    'colsample_bytree': [i/100.0 for i in range(75, 90 5)]
#}
