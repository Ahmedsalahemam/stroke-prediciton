import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
import warnings
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("healthcare-dataset-stroke-data.csv")
##print(data.isnull().sum())
## multinumericreplacing(data, 'smoking_status')
##normalization for data
##norm = StandardScaler()
##X_train_std = norm.fit_transform(first_train)
##X_test_std = norm.transform(first_test)

def preprocessin():
    ##removing nulls
    '''''
    Data.drop('id', axis=1, inplace=True)
    Data['gender'] = Data['gender'].replace({'Female': 0, 'Male': 1})
    Data['ever_married'] = Data['ever_married'].replace({'No': 0, 'Yes': 1})
    Data=Data.drop('work_type',axis=1)
    Data['Residence_type'] = Data['Residence_type'].replace({'Rural': 0, 'Urban': 1})
    '''
    ##transform data and assign the values to dataframe
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    enc = LabelEncoder()
    gender = enc.fit_transform(data['gender'])
    smoking_status = enc.fit_transform(data['smoking_status'])
    ##print(smoking_status)
    work_type = enc.fit_transform(data['work_type'])
    Residence_type = enc.fit_transform(data['Residence_type'])
    ever_married = enc.fit_transform(data['ever_married'])
    data['work_type'] = work_type
    data['ever_married'] = ever_married
    data['Residence_type'] = Residence_type
    data['smoking_status'] = smoking_status
    data['gender'] = gender

preprocessin()
y = data['stroke']
x = data.drop('stroke', axis=1)
def decision_tree():
    # Train-test split
    first_train, first_test, secnod_train, secnod_test = train_test_split(x, y, train_size=0.8,shuffle=True,random_state=2)

    ##decision tree for training
    dt = DecisionTreeClassifier()
    df=dt.fit(first_train,secnod_train).predict(first_test)

    ##dt.feature_importances_
    ## m = dt.predict(first_test)
    accuracyy = accuracy_score(secnod_test, df)
    print(1-accuracyy)
    ##print(Data)
def knn():
    f_train, f_test, s_train, s_test = train_test_split(x, y, train_size=0.8,shuffle=True,random_state=2)
    neigh = KNeighborsClassifier(n_neighbors=3)
    res = neigh.fit(f_train,s_train).predict(f_test)
    accuracyy = accuracy_score(s_test, res)
    print(1 - accuracyy)
def Naive_bayes():
    f_train, f_test, s_train, s_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=2)
    gs=GaussianNB()
    res=gs.fit(f_train,s_train).predict(f_test)
    accuracyy = accuracy_score(s_test, res)
    print(1 - accuracyy)









##main
decision_tree()
knn()
Naive_bayes()















