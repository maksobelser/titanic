import pandas as pd
import numpy as nu

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

import string
import warnings
warnings.filterwarnings('ignore')
seed = 42

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# podatki

df_test = pd.read_csv(r'D:\1_Faks\2_Uporabna statistika Msc\Strojno ucenje\DN\1_DN\tabular-playground-series-apr-2021\test.csv')
df_train = pd.read_csv(r'D:\1_Faks\2_Uporabna statistika Msc\Strojno ucenje\DN\1_DN\tabular-playground-series-apr-2021\train.csv')

df_train.name = 'Training Set'
df_test.name = 'Test set'

dfs = [df_train, df_test]
print('Stevilo enot za treniranje = {}'.format(df_train.shape[0]))
print('Stevilo enot za testiranje = {}'.format(df_test.shape[0]))
print('Stevilo spremenljivk za treniranje = {}'.format(df_test.shape[1]))
print(['Vse spremenljivke:', df_train.columns])
#
#
#
print(df_train.info())

# manjkajoče vrednosti

def display_missing(df):
    for col in df.columns.tolist():
        print('{} ima toliko manjkajočih vrednosti: {}'.format(col, df[col].isnull().sum()))
    print('\n')

for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)

df_train.columns.tolist()

# koralacije za dopolnitev manjkajočih vrednosti

df_train_corr = df_train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
print(df_train_corr[df_train_corr['Feature 1'] == 'Age'])

# histogrami za age

g = sns.FacetGrid(df_train, row="Pclass", col="Sex", margin_titles=True)
fig = g.map(sns.histplot, "Age")
plt.show()
# fig.savefig('histogrami.png')

# age

age_by_pclass_sex = df_train.groupby(['Sex', 'Pclass']).median()['Age']

# mediane

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mediana starosti po Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mediana starosti za vse: {}'.format(df_train['Age'].median()))

# zafilamo
df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# embarked, zelo malo vrednosti, spustim
(df_train['Embarked'].isnull().sum()/df_train.shape[0])*100

#  fare korelacije

print(df_train_corr[df_train_corr['Feature 1'] == 'Fare'])

g = sns.FacetGrid(df_train, row="Pclass", col="Sex", margin_titles=True)
fig_2 = g.map(sns.histplot, "Fare")
plt.show()
# fig_2.savefig('histogrami_fare.png')

fare_by_pclass_sex = df_train.groupby(['Sex', 'Pclass']).median()['Fare']
print(fare_by_pclass_sex)
fare_by_pclass_sex_mean = df_train.groupby(['Sex', 'Pclass']).mean()['Fare']
print(fare_by_pclass_sex_mean)

df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Fare'].apply(lambda x: x.fillna(x.mean()))

# Deck

df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

# graf

