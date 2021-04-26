import pandas as pd
import numpy as np

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

df_train['Fare'] = df_train.groupby(['Sex', 'Pclass'])['Fare'].apply(lambda x: x.fillna(x.mean()))

# Deck

df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

display_missing(df_train)

# ticket pustim, saj je v tako velikem podatkovju preveč zajebano

# nove znčilke

df_train['Fare'] = pd.qcut(df_train['Fare'], 13)
df_train['Age'] = pd.qcut(df_train['Age'], 10)

# kategorične spremenljivke

kat = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

for i, feature in enumerate(kat, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Število potnikov', size=20, labelpad=15)
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.legend(['Umrli', 'Preživeli'], loc='upper center', prop={'size': 18})
    plt.title('Smrtnost pri {}'.format(feature), size=20, y=0.995)

plt.show()

df_train['Druzina'] = df_train['SibSp'] + df_train['Parch'] + 1
family_map = {1: 'Sam', 2: 'Majhna', 3: 'Majhna', 4: 'Majhna', 5: 'Srednja', 6: 'Srednja', 7: 'Velika', 8: 'Velika', 11: 'Velika'}
df_train['Druzina_kat'] = df_train['Druzina'].map(family_map)

df_train['Priimek'] = df_train['Name'].str.split(', ', expand=True)[0]
100000/len(np.unique(df_train['Priimek'].tolist()))

# koliko % jih lahko pogrupiram pri ticket
1 - len(np.unique(df_train['Ticket'].tolist()))/(100000 - sum(df_train['Ticket'].isnull()))

#################################### MODEL

# kategoricne za model

ne_numericne = ['Embarked', 'Sex', 'Deck', 'Druzina_kat', 'Age', 'Fare']

for feature in ne_numericne:
    df_train[feature] = LabelEncoder().fit_transform(df_train[feature])

# one-hot encoding

kategoricne = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Druzina_kat']
encoded_features = []

for feature in kategoricne:
    encoded_feat = OneHotEncoder().fit_transform(df_train[feature].values.reshape(-1, 1)).toarray()
    n = df_train[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = df_train.index
    encoded_features.append(encoded_df)

df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)

drop_cols = ['Deck', 'Embarked', 'Druzina', 'Druzina_kat', 'Survived', 'Parch',
             'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Name', 'Priimek', 'Cabin']

df_train = df_train.drop(columns=drop_cols)

df_train.info()

X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))
y_train = df_train['Survived'].values

X_train.shape
y_train.shape

####### RANDOM FOREST

model_V1 = RandomForestClassifier(criterion='gini',
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)

leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)

N = 5
oob = 0
# probs = pd.DataFrame(np.zeros((len(X_train), N * 2)),
#                      columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
# importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)],
#                            index=df_train.columns)
# fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print('Fold {}\n'.format(fold))

    # Fitting the model
    leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])

    # Computing Train AUC score
    # trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx],
    #                                              leaderboard_model.predict_proba(X_train[trn_idx])[:, 1])
    # trn_auc_score = auc(trn_fpr, trn_tpr)
    # # Computing Validation AUC score
    # val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx],
    #                                              leaderboard_model.predict_proba(X_train[val_idx])[:, 1])
    # val_auc_score = auc(val_fpr, val_tpr)
    #
    # scores.append((trn_auc_score, val_auc_score))
    # fprs.append(val_fpr)
    # tprs.append(val_tpr)

    # X_test probabilities
    # probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]
    # probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]
    # importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_

    oob += leaderboard_model.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))

print('Average OOB Score: {}'.format(oob))

def model(model):
    N = 5
    napovedi = 0

    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # kje smo
        print('Fold {}\n'.format(fold))

        model.fit(X_train[trn_idx], y_train[trn_idx])

        napovedi += leaderboard_model.oob_score_ / N
        print('Fold {} Napoved: {}\n'.format(fold, model.oob_score_))

    print('Povprečna napoved: {}'.format(napovedi))

model(model_V1)

