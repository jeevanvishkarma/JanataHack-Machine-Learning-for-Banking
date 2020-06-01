import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum()


df = pd.concat([train,test], axis = 0)

df.isnull().sum()

train.Interest_Rate.value_counts()


# employee lenght

df['Length_Employed'].replace({'< 1 year' : '0','1 year': '1','10+ years' : '10'}, inplace = True)
df['Length_Employed'] = df['Length_Employed'].str.replace('years','')

df['Length_Employed'].fillna(-1, inplace = True)
df['Length_Employed'] = df['Length_Employed'].astype(int)

df['Length_Employed'].value_counts()
sns.distplot(df['Length_Employed'])


#df['Length_Employed'] = pd.cut(df['Length_Employed'], bins = [-2,-1, 1,3,6,9,11], labels = [-1,1,2,3,4,10])

df['Length_Employed'].value_counts()


# HOme owned

sns.countplot(df.Home_Owner)
df.Home_Owner  = df.Home_Owner.map(lambda x : 3 if x == 'Own' else 2 if x == 'Mortgage'
                                   else 1 if x == 'Rent' else   -1)
df.Home_Owner.value_counts()
# loan amount
df.Loan_Amount_Requested = df.Loan_Amount_Requested.str.replace(',','').astype(int)

df.Loan_Amount_Requested = np.cbrt(df.Loan_Amount_Requested)

sns.boxplot(df.Interest_Rate,df.Loan_Amount_Requested)


sns.distplot(df.Loan_Amount_Requested)


# income verified

sns.countplot(df.Income_Verified)
df.Income_Verified = df.Income_Verified.map(lambda x : 0 if x == 'not verified'
                                                else 3 if x == 'VERIFIED - income' else 2)

# Purpose
sns.countplot(df.Purpose_Of_Loan)
df.Purpose_Of_Loan.value_counts()
pd.crosstab(df.Purpose_Of_Loan, df.Interest_Rate).plot.bar()

df.Purpose_Of_Loan = df.Purpose_Of_Loan.map(lambda x : 5 if x == 'debt_consolidation'
                                            else 4 if x == 'credit_card'
                                            else 3 if x == 'home_improvement'
                                            else 2 if x == 'other'
                                            else -1)
df.Purpose_Of_Loan.astype(int)

# debt

df.Debt_To_Income.describe()

sns.distplot(df.Debt_To_Income)

# Inquiries_Last_6Mo

sns.countplot(df.Inquiries_Last_6Mo)
df.Inquiries_Last_6Mo.value_counts()

df.Inquiries_Last_6Mo = df.Inquiries_Last_6Mo.map(lambda x : 3 if x >= 3 else x)

#Months_Since_Deliquency

sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.describe()

def IQR(df,variable,distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    upper = df[variable].quantile(0.75) + (IQR*distance)
    lower = df[variable].quantile(0.25) - (IQR*distance)
    return lower, upper
 
IQR(df, 'Months_Since_Deliquency',1.5)

df.Months_Since_Deliquency = df.Months_Since_Deliquency.map(lambda x : 95 if x > 95 else x)
sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.fillna(0, inplace = True)

df.Months_Since_Deliquency = np.cbrt(df.Months_Since_Deliquency)
df.Months_Since_Deliquency.isna().sum()

df.Months_Since_Deliquency.describe()

sns.countplot(df.Months_Since_Deliquency)
pd.crosstab(df.Months_Since_Deliquency,df.Interest_Rate)


#Total_Accounts
sns.distplot(df.Total_Accounts)
df.Total_Accounts.describe()

IQR(df, 'Total_Accounts', 1.5)

df.Total_Accounts= df.Total_Accounts.map(lambda x : 58 if x > 58 else x)

df.Total_Accounts =np.sqrt(df.Total_Accounts)




#df.Total_Accounts = pd.cut(df.Total_Accounts, bins = [0,8,12,18,24,30,36,42,48,160],
                    #  labels = [1,2,3,4,5,6,7,8,9])
#Number_Open_Accounts

sns.distplot(df.Number_Open_Accounts)

IQR(df,'Number_Open_Accounts', 1.5)
import math
df.Number_Open_Accounts = df.Number_Open_Accounts.map(lambda x : 25 if x > 25 else x)

#df.Number_Open_Accounts =  pd.cut(df.Number_Open_Accounts, bins = [-1,4,6,8,10,12,14,16,18,20,86],
               #     labels = [1,2,3,4,5,6,7,8,9,10])

df.Number_Open_Accounts = np.sqrt(df.Number_Open_Accounts)
df.Number_Open_Accounts.value_counts()

a = np.cbrt(df.Number_Open_Accounts)
sns.distplot(a)


# Closed account

df['closed_accounts']= df['Total_Accounts'] - df['Number_Open_Accounts']

IQR(df, 'closed_accounts', 1.5)


df.closed_accounts.describe()

sns.distplot(df.closed_accounts)

# gender 

df.Gender.replace({'Female' : 1, 'Male' : 0}, inplace = True)

# Annual_Income

df.Annual_Income.describe().apply(lambda x: format(x, 'f'))
df.Annual_Income.median()
IQR(df,'Annual_Income', 1.5)

df.Annual_Income.describe()

df.Annual_Income = df.Annual_Income.map(lambda x :155000 if x >155000 else x)

df.isna().sum()
df.Annual_Income = df.groupby('Debt_To_Income')['Annual_Income'].transform(lambda x: x.fillna(x.mean()))
sns.distplot(df.Annual_Income)

df.Annual_Income.describe()
df.isnull().sum()

df = df.drop(['Loan_ID','Gender'], axis = 1)

# model
training = df.iloc[:len(train_under),:]
testing = df.iloc[len(train_under):, : ]
testing = testing.drop('Interest_Rate', axis = 1)

# Model Creation

X = training.drop('Interest_Rate', axis = 1)
y = training['Interest_Rate']

from sklearn.model_selection import train_test_split

x_train, x_val,y_train,y_val = train_test_split(X,y, test_size = 0.18, random_state = 21)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from lightgbm import LGBMClassifier

class_1 =( len(y) - len(y[y==1]))/len(y)
class_2 =( len(y) - len(y[y==2]))/len(y)
class_3 =( len(y) - len(y[y==3]))/len(y)

weight = {1: '0.79', 2: '0.65', 3: '0.68'}

lgb =LGBMClassifier(boosting_type='gbdt',
                       max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       class_weight = weight,
                       min_child_weight = 0.02,
                       colsample_bytree=0.6, 
                       random_state=7,
                       objective='multiclass')

lgb.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_val, y_val.values)],
          early_stopping_rounds=500,
          verbose=200)


print(lgb.feature_importances_)

plt.bar(X.columns, lgb.feature_importances_)
plt.xticks(rotation = 90)


print(accuracy_score(y_val,lgb.predict(x_val)))

submission = pd.DataFrame()
submission['Loan_ID'] = test['Loan_ID']
submission['Interest_Rate'] = lgb.predict(testing)
submission.to_csv('1.csv', index=False, header=True)


df.hist(figsize = (12,12),density = True)
plt.show()
# kfold
from sklearn.model_selection import StratifiedKFold

err = []
y_pred_tot_lgm = []

from sklearn.model_selection import StratifiedKFold

fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)
i = 1
for train_index, test_index in fold.split(X, y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    m = LGBMClassifier(boosting_type='gbdt',
                       max_depth=7,
                       learning_rate=0.05,
                       n_estimators=5000,
                       colsample_bytree=0.7,
                       random_state=1994,
                       objective='multiclass')
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=200,
          verbose=200)
    pred_y = m.predict(x_val)
    print(i, " err_lgm: ", accuracy_score(y_val, pred_y))
    err.append(accuracy_score(y_val, pred_y))
    pred_test = m.predict(testing)
    i = i + 1
    y_pred_tot_lgm.append(pred_test)
    
np.mean(err,0)

prediction = pd.DataFrame()
for i in range(0, 15):
    prediction = pd.concat([prediction,pd.DataFrame(y_pred_tot_lgm[i])],axis=1)


submission = pd.DataFrame()
submission['Loan_ID'] = test['Loan_ID']
submission['Interest_Rate']=prediction.mode(axis=1)[0]
submission.to_csv('mode.csv', index=False, header=True)










