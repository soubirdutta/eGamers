import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import seaborn as sns
 
%matplotlib inline

df = pd.read_csv('Data.csv')

#DATA CLEANING

df = df.drop(['S. No.', 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7', 'GADE', 'Timestamp', 'SWL1', 'SWL2', 'SWL3', 
               'SWL4', 'SWL5', 'Game', 'League', 'highestleague', 'streams', 'SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5', 
               'SPIN6', 'SPIN7', 'SPIN8', 'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12', 'SPIN13', 'SPIN14', 'SPIN15', 'SPIN16', 
               'SPIN17', 'Narcissism', 'Reference', 'accept', 'SWL_T', 'SPIN_T', 'Birthplace', 'Residence',
               'Birthplace_ISO3', 'Residence_ISO3', 'earnings'], axis=1) #Birthplace/Residence cleaned up is ISO3

df['GAD_T'] = df['GAD_T'].apply(lambda x: x/7) #averaging GAD Total score between 0 and 3
df['GAD_T'] = df['GAD_T'].apply(lambda x: 0 if x <= 1.0 else 1) #putting players into buckets of at-risk(0) or not at-risk(1)
df.rename(columns={'GAD_T': 'status'}, inplace=True) #renaming to more relevant name

df['Platform'] = df['Platform'].map({'Console (PS, Xbox, ...)': 1, 'PC': 2, 'Smartphone / Tablet': 3}) #Mapping devices to int

df['whyplay'] = df['whyplay'].map({'having fun': 1, 'improving': 2, 'relaxing': 3, 'winning': 4}) #Mapping motivation to int

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2, 'Other': 3}) # Mapping gender to int values

df['Work'] = df['Work'].map({'Unemployed / between jobs': 1, 'Employed': 2, 'Student at college / university': 3,
                             'Student at school': 4}) # Mapping work to int values

df['Degree'] = df['Degree'].map({'Bachelor�(or equivalent)' : 3, 'High school diploma (or equivalent)': 2,
       'Ph.D., Psy. D., MD (or equivalent)': 4, 'Master�(or equivalent)': 4, 'None': 1}) # Mapping educational degree to int values

df['Playstyle'] = df['Playstyle'].map({'Singleplayer': 1,
                                       'Multiplayer - online - with strangers': 2, 
                                       'Multiplayer - online - with online acquaintances or teammates': 3, 
                                       'Multiplayer - online - with real life friends': 4, 
                                       'Multiplayer - offline (people in the same room)': 5}) # Mapping playing environment to int values

df = df.dropna() #Dropping nan values

#EXPLORATORY DATA ANALYSIS

#Graph of age distribution in dataset:
tempdf = df['Age'].value_counts()
tempdf = pd.DataFrame(tempdf)
n = tempdf.sum(axis=0)
tempdf['Percent'] = tempdf['Age'].apply(lambda x: (x*100)/n)
sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
sns.countplot(x='Age', data=df)
plt.ylabel('Count of Respondents')

#Heatmap:
plt.figure(figsize=(12,4))
df_heatmap = df.corr()
sns.heatmap(data=df_heatmap, annot=True)

#Graph of Hours played vs -ve effects
# Cutting playing hours per week into bins (as there were 85 unique values)
bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
group_names = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50+']
df['Hour Bins'] = pd.cut(df['Hours'], bins, labels=group_names)
tempdf = df.groupby('Hour Bins')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
plt.figure(figsize=(12,4))
sns.barplot(x='Hour Bins', y='Percent', data=tempdf)
plt.xlabel('Hours Played per Week')
plt.ylabel('% of At-Risk Players')

#Graph of WhyPlay vs -ve effects:
tempdf = df.groupby('whyplay')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
tempdf['whyplay'] = tempdf['whyplay'].map({1: 'Fun', 2: 'Improvement', 3: 'Relaxing', 4: 'Winning'}) # Unmapping for graph
tempdf = tempdf[tempdf['whyplay'] != 'Relaxing'] # removing 'Relaxing due to small sample size compared to others (see above cell)
plt.figure(figsize=(10,4))
sns.barplot(x='whyplay', y='Percent', data=tempdf)
plt.xlabel('Reason for Playing')
plt.ylabel('% of At-Risk Players')

#Graph of Age vs -ve effects:
tempdf = df.groupby('Age')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
tempdf = tempdf[tempdf['Age'] <= 27] # to ensure sample size of min 200 respondents (see above cell)
plt.figure(figsize=(10,4))
sns.barplot(x='Age', y='Percent', data=tempdf)
plt.xlabel('Age')
plt.ylabel('% of At-Risk Players')

#Graph of Work vs -ve effects:
tempdf = df.groupby('Work')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
tempdf['Work'] = tempdf['Work'].map({1: 'Unemployed / between jobs', 2: 'Employed', 3: 'Student (school or college)',
                             4: 'Student (school or college)'}) # Unmapping for graph...some error on re-running
plt.figure(figsize=(8,6))
sns.barplot(x='Work', y='Percent', data=tempdf)
plt.xlabel('Work Status')
plt.ylabel('% of At-Risk Players')

#Graph of Degree vs -ve effects:
tempdf = df.groupby('Degree')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
# Unmapping for graph // Combine 3 and 4 due to low sample of 4 - ToDo..
tempdf['Degree'] = tempdf['Degree'].map({3: 'Bachelors', 2: 'High School', 1: 'None', 4: 'Masters or Higher'}) 
plt.figure(figsize=(10,4))
sns.barplot(x='Degree', y='Percent', data=tempdf)
plt.xlabel('Educational Degree')
plt.ylabel('% of At-Risk Players')

#Graph of Playstyle vs -ve effects:
tempdf = df.groupby('Playstyle')['status'].value_counts(normalize=True) # New temp df with containing only relevant values
tempdf = tempdf.mul(100).rename('Percent').reset_index() # Multiplying values by 100 as 'normalize' sets value between 0 & 1
tempdf = tempdf[tempdf['status'] == 1] # tempdf to include only players where status = 1 (ie those who have anxiety)
tempdf = tempdf[tempdf['Playstyle'] != 5.0] # Dropping offline multi-player due to low sample size
# Unmapping for graph:
tempdf['Playstyle'] = tempdf['Playstyle'].map({1: 'Singleplayer',
                                               2: 'Multiplayer (with strangers)', 
                                               3: 'Multiplayer (with online acquaintances)', 
                                               4: 'Multiplayer (with real life friends)'})
tempdf.head()
plt.figure(figsize=(14,4))
sns.barplot(x='Playstyle', y='Percent', data=tempdf)
plt.xlabel('Playing Style')
plt.ylabel('% of At-Risk Players')

#LOGISTIC REGRESSION MODEL

#Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1),
                                                    df['status'], test_size=0.30,
                                                    random_state=101)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('status',axis=1))
scaled_features = scaler.transform(df.drop('status',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

#Training
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))

#Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,rfc_pred))
print('\n')
print(confusion_matrix(y_test,rfc_pred))
