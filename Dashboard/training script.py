import numpy as np
import pandas as pd

import seaborn as sns
import category_encoders as ce

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS

import pickle

bankloan= pd.read_csv('data.csv')

ordinal_mapping=[{'col':'Education','mapping':{None:0,'Undergrad':1,'Graduate':2, 'Advanced/Professional':3}}]
ordinal_encoder=ce.OrdinalEncoder(mapping=ordinal_mapping)
bankloan_ord=ordinal_encoder.fit_transform(bankloan['Education'])

transformer= ColumnTransformer([('one_hot',OneHotEncoder(drop='first'),['Securities Account', 'CD Account', 'Online', 'CreditCard']),
('binary',ce.BinaryEncoder(),['Education'])
], remainder='passthrough')

fitur=['Age','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']
X=bankloan[fitur]
y=[1 if i=='Yes' else 0 for i in bankloan['Personal Loan']]

model=RandomForestClassifier()
under=NearMiss()
over=SMOTE()

estimator = Pipeline([
    ('preprocess', transformer), ('balance', under),
    ('model', model)])

skfold=StratifiedKFold(n_splits=5)

hyperparam_space={'balance':[over,under]}

grid_search= GridSearchCV(estimator,
    param_grid= hyperparam_space,
    cv=skfold,
    n_jobs=-1,
    scoring='f1')

grid_search.fit(X,y)

grid_search.best_estimator_.fit(X,y)
file_name='Model_Final.sav'

pickle.dump(grid_search.best_estimator_,open(file_name,'wb'))