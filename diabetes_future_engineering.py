from helper_functions import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from scipy.stats import norm, skew
from scipy import stats

df = pd.read_csv('datasets/diabetes.csv')

###################################
# Understanding the data
###################################
check_df(df)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# for col in cat_cols:
#     target_summary_with_cat(df, 'Outcome', col)

for col in num_cols:
    target_summary_with_num(df, 'Outcome', col, plot=True)

###################################
# Outliers
###################################

for col in num_cols:
    print(col, ':', check_outlier(df, col))

for col in num_cols:
    grab_outliers(df, col)

for col in num_cols:
    boxplot_outliers(df, col)

###################################
# Missing values
###################################

missing_values_table(df)

df_corr(df)

new_num_cols = num_cols.copy()
new_num_cols.remove('Pregnancies')
new_num_cols.remove('DiabetesPedigreeFunction')

for col in new_num_cols:
    if len(df[df[col] == 0].index) != 0:
        df[col] = df[col].replace(0, np.nan)

df.head()

df.isnull().sum()

na_cols = missing_values_table(df, True)

missing_vs_target(df, 'Outcome', na_cols)

##################
# Adding new features
##################

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 35), 'NEW_AGE_CAT'] = 'middle_age'
df.loc[(df['Age'] >= 35) & (df['Age'] < 55), 'NEW_AGE_CAT'] = 'above_middle_age'
df.loc[(df['Age'] >= 55), 'NEW_AGE_CAT'] = 'old'

df['Glucose'].mean()
df.groupby(['NEW_AGE_CAT'])['Glucose'].mean()
df['Glucose'] = df['Glucose'].fillna(df.groupby('NEW_AGE_CAT')['Glucose'].transform('mean'))

# df.groupby(['NEW_GLUCOSE_140'])['Insulin'].mean()
df['Insulin'].mean()
df.groupby(['NEW_AGE_CAT'])['Insulin'].mean()
df['Insulin'] = df['Insulin'].fillna(df.groupby('NEW_AGE_CAT')['Insulin'].transform('mean'))

df.loc[(df['Glucose'] > 140), 'NEW_GLUCOSE_140'] = 'YES'
df.loc[(df['Glucose'] <= 140), 'NEW_GLUCOSE_140'] = 'NO'

df['NEW_GLUCOSE_INSULIN'] = df['Glucose'] * df['Insulin']

df.loc[(df['NEW_GLUCOSE_INSULIN']/405 < 25), 'HOMA_IR'] = 'NO'
df.loc[(df['NEW_GLUCOSE_INSULIN']/405 >= 25), 'HOMA_IR'] = 'YES'

###################################
# Filling the missing values
###################################

df['BloodPressure'].mean()
df.groupby(['NEW_AGE_CAT'])['BloodPressure'].mean()
df['BloodPressure'] = df['BloodPressure'].fillna(df.groupby('NEW_AGE_CAT')['BloodPressure'].transform('mean'))

df['SkinThickness'].mean()
df.groupby(['NEW_AGE_CAT'])['SkinThickness'].mean()
df['SkinThickness'] = df['SkinThickness'].fillna(df.groupby('NEW_AGE_CAT')['SkinThickness'].transform('mean'))

df['BMI'].mean()
df.groupby(['NEW_GLUCOSE_140'])['BMI'].mean()
df['BMI'] = df['BMI'].fillna(df.groupby('NEW_GLUCOSE_140')['BMI'].transform('mean'))

##################
# New features
##################

df.loc[(df['BMI'] >= 30)  & (df['Age'] < 18), 'NEW_AGE_OBESE_CAT'] = 'young_obese'
df.loc[(df['BMI'] >= 30)  & (df['Age'] >= 18) & (df['Age'] < 35), 'NEW_AGE_OBESE_CAT'] = 'middle_age_obese'
df.loc[(df['BMI'] >= 30)  & (df['Age'] >= 35) & (df['Age'] < 55), 'NEW_AGE_OBESE_CAT'] = 'above_middle_age_obese'
df.loc[(df['BMI'] >= 30)  & (df['Age'] >= 55), 'NEW_AGE_OBESE_CAT'] = 'old_obese'

df.loc[(df['BMI'] < 30)  & (df['Age'] < 18), 'NEW_AGE_OBESE_CAT'] = 'young_normal'
df.loc[(df['BMI'] < 30)  & (df['Age'] >= 18) & (df['Age'] < 35), 'NEW_AGE_OBESE_CAT'] = 'middle_age_normal'
df.loc[(df['BMI'] < 30)  & (df['Age'] >= 35) & (df['Age'] < 55), 'NEW_AGE_OBESE_CAT'] = 'above_middle_age_normal'
df.loc[(df['BMI'] < 30)  & (df['Age'] >= 55), 'NEW_AGE_OBESE_CAT'] = 'old_normal'

df.loc[(df['BMI'] < 18.5), 'NEW_OBESE_CAT'] = 'underweight'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 24.9), 'NEW_OBESE_CAT'] = 'normal'
df.loc[(df['BMI'] >= 24.9) & (df['BMI'] < 30), 'NEW_OBESE_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 35), 'NEW_OBESE_CAT'] = 'obese'
df.loc[(df['BMI'] >= 35), 'NEW_OBESE_CAT'] = 'extra_obese'

df.loc[(df['BMI'] >= 35), 'NEW_OBESE'] = 'YES'
df.loc[(df['BMI'] < 35), 'NEW_OBESE'] = 'NO'

df['NEW_GLUCOSE_BP'] = df['Glucose'] / df['BloodPressure']

df['NEW_GLUCOSE_DPF'] = df['Glucose'] / df['DiabetesPedigreeFunction']

df['NEW_GLUCOSE_BMI'] = df['Glucose'] * df['BMI']

df.head()
df.shape

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

###################################
# Visualisations
###################################

for col in num_cols:
    plt.figure()
    sns.distplot(df[col], fit = norm)
    plt.show(block=True)

for col in num_cols:
    plt.figure()
    stats.probplot(df[col], plot = plt)
    plt.title(col)
    plt.show(block=True)

###################################
# Outliers and thresholds
###################################

for col in num_cols:
    print(col, ':', check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

###################################
# Visualisations
###################################

for col in num_cols:
    plt.figure()
    sns.distplot(df[col], fit = norm)
    plt.show(block=True)

for col in num_cols:
    plt.figure()
    stats.probplot(df[col], plot = plt)
    plt.title(col)
    plt.show(block=True)

###################################
# Encoding
###################################

missing_values_table(df)

df.isnull().sum()

binary_cols = binary_cols(df)

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# rare_analyser(df, 'Outcome', cat_cols)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.shape

#############################################
# Model
#############################################
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

# Veri setini eğitim ve test setleri olarak bölün.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

from helper_functions import *

###################################
# RFC Model estimator
###################################

scores =[]
for k in range(1, 120):
    rfc = RandomForestClassifier(n_estimators=k, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(range(1, 120), scores)
plt.xlabel('Value of n_estimators for Random Forest Classifier')
plt.ylabel('Testing Accuracy')
plt.show(block=True)

# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test, rfc_pred)
print('RandomForestClassifier accuracy:', rfc_acc) # 0.7987012987012987

# LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=1000,random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print('LogisticRegression accuracy:', lr_acc) # 0.7727272727272727

# GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
gbc_acc = accuracy_score(y_test, gbc_pred)
print('GradientBoostingClassifier accuracy:', gbc_acc) # 0.7532467532467533

# XGBClassifier
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05)
xgb_model.fit(X_train, y_train)
xgb_acc = xgb_model.score(X_test, y_test)
print('XGBClassifier accuracy:', xgb_acc) # 0.7727272727272727

###################################
# Model importance
###################################

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0:len(X)])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rfc, X_train)
plot_importance(gbc, X_train)
plot_importance(xgb_model, X_train)

ML_Data = pd.DataFrame(columns = ['Model_Name', 'Accuracy'])

Model_Name = ['RFC', 'LR', 'GBC','XGB']
Accuracy = [rfc_acc, lr_acc, gbc_acc, xgb_acc]

ML_Data['Model_Name'] = Model_Name
ML_Data['Accuracy'] = Accuracy

ax = sns.barplot(x=ML_Data['Model_Name'], y=ML_Data['Accuracy'])
ax.set_ylim([0.7, max(Accuracy)])
plt.xlabel('Model Name')
plt.ylabel('Accuracy Score')
plt.show(block=True)
